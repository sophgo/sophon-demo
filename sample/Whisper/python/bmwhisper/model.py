#===----------------------------------------------------------------------===#
#
# Copyright (C) 2024 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import os
import time
import base64
import gzip
from dataclasses import dataclass
import sophon.sail as sail
import numpy as np
import torch

from .decoding import decode as decode_function
from .decoding import detect_language as detect_language_function
from .transcribe import transcribe as transcribe_function
from .utils import fp16_cast, uint16_to_fp16


@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


class Whisper():
    def __init__(self, dims: ModelDimensions, args):
        self.version = "1.0.0"
        super().__init__()
        self.dims = dims
        self.encoder = None
        self.decoder = None
        self.encoder_infer = None
        self.logits_decoder_infer = None
        self.decoder_main_infer = None
        self.decoder_post_infer = None
        self.decoder_loop_infer = None
        self.model_name = args["model_name"]
        self.bmodel_dir = args["bmodel_dir"]
        self.beam_size = args["beam_size"]
        self.padding_size = args["padding_size"]
        self.dev_id = args["dev_id"]
        self.num_core = args["num_core"]
        self.stream = sail.nn.Stream(self.dev_id)

        # use the last half layers for alignment by default; see `set_alignment_heads()` below
        all_heads = torch.zeros(
            self.dims.n_text_layer, self.dims.n_text_head, dtype=torch.bool
        )
        all_heads[self.dims.n_text_layer // 2 :] = True
        self.alignment_heads = all_heads.to_sparse()

        # get positional embedding from npz file
        positional_embedding_path = os.path.join(os.path.dirname(__file__), "assets", f"positional_embedding_{self.model_name}.npz")
        assert os.path.exists(positional_embedding_path), f"{positional_embedding_path} not found"
        self.positional_embedding = torch.tensor(np.load(positional_embedding_path)["positional_embedding"])

        ########################################
        ## Using sail to load BModel
        ########################################
        start_time = time.time()
        combined_whisper_model_path = f"bmwhisper_{self.model_name}_1690_f16_{self.num_core}core.bmodel"
        combined_whisper_model_path = os.path.join(self.bmodel_dir, combined_whisper_model_path)
        assert os.path.exists(combined_whisper_model_path), f"{combined_whisper_model_path} not found"

        # load combined model
        self.combined_whisper_engine = sail.nn.Engine(combined_whisper_model_path, self.dev_id)

        net_names =  self.combined_whisper_engine.get_net_names()
        graph_names_raw = ["encoder", "logits_decoder", "decoder_main", "decoder_post", "decoder_loop", "kvcache_rearrange"]
        graph_names = []
        for i in range(len(net_names)):
            for j in range(len(graph_names_raw)):
                if graph_names_raw[j] in net_names[i]:
                    graph_names.append(graph_names_raw[j])
        for i in range(len(net_names)):
            setattr(self, f"{graph_names[i]}_graph_name", net_names[i])
            input_names = self.combined_whisper_engine.get_input_names(net_names[i])
            setattr(self, f"{graph_names[i]}_input_names", input_names)
            output_names = self.combined_whisper_engine.get_output_names(net_names[i])
            setattr(self, f"{graph_names[i]}_output_names", output_names)
            input_tensors_map = self.create_input_tensor_map(self.combined_whisper_engine,
                                                             net_names[i],
                                                             stage_id=0,
                                                             dev_id=self.dev_id)
            setattr(self, f"{graph_names[i]}_input_tensors_map", input_tensors_map)
            output_tensors_map = self.create_output_tensor_map(self.combined_whisper_engine,
                                                               net_names[i],
                                                               stage_id=0,
                                                               dev_id=self.dev_id)
            setattr(self, f"{graph_names[i]}_output_tensors_map", output_tensors_map)
        self.kvcache_rearrange_input_list = []
        self.kvcache_rearrange_output_list = []
        for i in range(self.dims.n_text_layer * 2):
            # initial kvcache_rearrange engine
            kvcache_rearrange_input_tensors_map = self.kvcache_rearrange_input_tensors_map
            kvcache_rearrange_output_tensors_map = self.kvcache_rearrange_output_tensors_map
            
            kvcache_rearrange_input_tensors_map[0] = self.decoder_main_output_tensors_map[i + 1]

            kvcache_rearrange_output_tensors_map[0] = kvcache_rearrange_input_tensors_map[0]

            self.kvcache_rearrange_input_list.append(kvcache_rearrange_input_tensors_map)
            self.kvcache_rearrange_output_list.append(kvcache_rearrange_output_tensors_map)

        for i in range(self.dims.n_text_layer * 4):
            self.decoder_loop_input_tensors_map[i + 3] = self.decoder_main_output_tensors_map[i + 1]

        for i in range(self.dims.n_text_layer * 2):
            self.decoder_loop_output_tensors_map[i + 1] = self.decoder_loop_input_tensors_map[i + 3]

        kvcache_rearrange_engine_base_input = self.kvcache_rearrange_input_list[0][1]
        for i in range(self.dims.n_text_layer * 2 - 1):
            self.kvcache_rearrange_input_list[i + 1][1] = kvcache_rearrange_engine_base_input

        model_init_time = time.time() - start_time
        print(f"\nTPU bmodel init time: {model_init_time}s")

        self.inference_time = 0
        self.preprocess_time = 0
        self.postprocess_time = 0
        self.main_loop_cnt = 0
        self.call_encoder = 0
        self.call_logits_decoder= 0
        self.call_decoder_loop= 0
        self.call_decoder_firstly= 0
        self.call_decoder_with_kvcache = 0
        self.call_kvcache_rearrange = 0
        self.max_ctx = 0
    
    def create_input_tensor_map(self, net, net_name, stage_id=0, dev_id=0):
        input_dtypes = net.get_input_dtypes(net_name)
        input_shapes = net.get_input_shapes(net_name, stage_id)
        input_tensors_map = {}
        for i in range(len(input_shapes)):
            input_tensors_map[i] = sail.nn.Tensor(input_shapes[i], input_dtypes[i], dev_id)
        return input_tensors_map
    
    def create_output_tensor_map(self, net, net_name, stage_id=0, dev_id=0):
        output_dtypes = net.get_output_dtypes(net_name)
        output_shapes = net.get_output_shapes(net_name, stage_id)
        output_tensors_map = {}
        for i in range(len(output_shapes)):
            output_tensors_map[i] = sail.nn.Tensor(output_shapes[i], output_dtypes[i], dev_id)
        return output_tensors_map

    def init_time(self):
        self.inference_time = 0
        self.preprocess_time = 0
        self.postprocess_time = 0

    def init_cnt(self):
        self.main_loop_cnt = 0
        self.call_encoder = 0
        self.call_logits_decoder= 0
        self.call_decoder_loop= 0
        self.call_decoder_firstly= 0
        self.call_kvcache_rearrange = 0

    def print_cnt(self):
        def print_cnt(text, cnt, n):
            print(f"{text:<{n}} {cnt}")
        print()
        print_cnt("Call encoder times:", self.call_encoder, 50)
        print_cnt("Call logits decoder times:", self.call_logits_decoder, 50)
        print_cnt("Call decoder firstly times:", self.call_decoder_firstly, 50)
        print_cnt("Call decoder loop:", self.call_decoder_loop, 50)
        print_cnt("Call kvcache rearrange times:", self.call_kvcache_rearrange, 50)

    def set_alignment_heads(self, dump: bytes):
        array = np.frombuffer(
            gzip.decompress(base64.b85decode(dump)), dtype=bool
        ).copy()
        mask = torch.from_numpy(array).reshape(
            self.dims.n_text_layer, self.dims.n_text_head
        )
        self.alignment_heads = mask.to_sparse()

    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        # hard code tokens type here
        tokens = tokens.numpy().astype(np.int32)
        audio_features = audio_features.numpy().astype(np.float16)
        tokens = tokens if tokens.flags.c_contiguous else np.ascontiguousarray(tokens)
        audio_features = audio_features if audio_features.flags.c_contiguous else np.ascontiguousarray(audio_features)

        self.logits_decoder_input_tensors_map[0].copy_(tokens)
        unint16_audio_features = fp16_cast(audio_features)
        self.logits_decoder_input_tensors_map[1].copy_(unint16_audio_features);

        self.combined_whisper_engine.process(self.logits_decoder_input_tensors_map, 
                                             self.logits_decoder_output_tensors_map, 
                                             self.stream, 
                                             self.logits_decoder_graph_name)
        logits_tensor = list(self.logits_decoder_output_tensors_map.values())[0]
        logits_tensor.to_('host')
        logits = torch.from_numpy(uint16_to_fp16(logits_tensor.asnumpy()))
        logits_tensor.to_('device')

        self.call_logits_decoder += 1
        return logits

    @property
    def is_multilingual(self):
        return self.dims.n_vocab >= 51865

    @property
    def num_languages(self):
        return self.dims.n_vocab - 51765 - int(self.is_multilingual)

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = decode_function
