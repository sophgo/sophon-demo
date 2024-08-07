#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional, Sequence, Set, final
# import librosa
import ffmpeg
from typing import List
import sophon.sail as sail
import logging
# import ast
import math

from fairseq2.data.audio import WaveformToFbankConverter, WaveformToFbankInput
from fairseq2.models.nllb.tokenizer import NllbTokenizer
from fairseq2.data.data_pipeline import Collater
from fairseq2.nn.padding import get_seqs_and_padding_mask

from fairseq2.data.text import SentencePieceEncoder, SentencePieceTokenizerBase
from fairseq2.data.typing import PathLike
from fairseq2.typing import Device, finaloverride
from fairseq2.nn.position_encoder import SinusoidalPositionEncoder
logging.basicConfig(level=logging.INFO)
# sail.set_print_flag(1)

# standarized audio
def buf_to_float(x, *, n_bytes=2, dtype=np.float32):
    scale = 1.0 / float(1 << ((8 * n_bytes) - 1))
    fmt = "<i{:d}".format(n_bytes)
    return scale * np.frombuffer(x, fmt).astype(dtype)

class SileroVADSilenceRemover:
    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        self.model, self.utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            # force_reload=True,
            onnx=False,
        )

    def __call__(self, sample: torch.Tensor, is_standardized: bool) -> List[float]:
        if not is_standardized:
            # Standardizing here just for getting silence boundaries
            standarized_sample_list = F.layer_norm(sample, sample.shape).tolist()
        else:
            standarized_sample_list = sample.tolist()

        (
            get_speech_timestamps,
            save_audio,
            read_audio,
            VADIterator,
            collect_chunks,
        ) = self.utils
        speech_timestamps = get_speech_timestamps(
            standarized_sample_list, self.model, sampling_rate=self.sample_rate
        )

        sample_list: List[float] = sample.tolist()
        if len(speech_timestamps) == 0:
            return sample_list
        speech_start_time = speech_timestamps[0]["start"]
        speech_end_time = speech_timestamps[-1]["end"]
        return sample_list[int(speech_start_time) : int(speech_end_time)]

class seamless_stream_s2tt:
    def __init__(self, args):
        self.silence_remover = SileroVADSilenceRemover()
        self.is_standardized = False
        self.chunk_duration_ms = args.chunk_duration_ms # NOTE: original code: 320
        self.tgt_lang = args.tgt_lang
        self.handle = sail.Handle(args.dev_id)
        self.dev_type = self.handle.get_target()
        assert self.dev_type in ['BM1684X', 'BM1688'], "only support BM1684X and BM1688 devices"
        self.use_slience_remover = args.use_slience_remover

        if self.dev_type == "BM1688":
            self.pos = SinusoidalPositionEncoder(1024, 4096, _legacy_pad_idx=1, device="cpu")

        # NOTE: the value effect the final result, meaning the number of segments, the value * self.chunk_duration_ms * 16 + self.previous_residual_samples don't exceed self.unity_max_encoder_frontend_input_length. 
        # the longer the value is, the richer the context is, the higher the precision might be, the more the compute cost is 
        self.consecutive_segments_num = args.consecutive_segments_num

        # OnlineFeatureExtractorAgent
        self.convert_to_fbank = WaveformToFbankConverter(
            num_mel_bins=80,
            waveform_scale=2**15 if False else 1.0,
            standardize=False,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.previous_residual_samples = []
        self.cur_input = []
        self.window_size = 25 # 50 # NOTE: original code: 25
        # NOTE: adaptively change by input, original code default param: 16000
        self.sample_rate = args.sample_rate 
        self.num_samples_per_window = int(self.window_size * self.sample_rate / 1000)
        self.len_ms_to_samples = lambda x: x * self.sample_rate / 1000
        self.shift_size = 10 # 20 # # NOTE: original code: 10
        self.num_samples_per_shift = int(self.shift_size * self.sample_rate / 1000)

        # OfflineWav2VecBertEncoderAgent
        self.min_starting_wait = args.fbank_min_starting_wait # 96 # 192
        self.min_input_length = args.fbank_min_input_length # 2
        # self.collate = Collater(
        #     pad_value=0, pad_to_multiple=2 # pad_value=text_tokenizer.vocab_info.pad_idx, pad_to_multiple=2
        # )
        self.pad_value = 0
        self.stride = 2
        self.adapter_kernel_size = 8
        self.adapter_stride = 8
        self.adapter_pad = self.adapter_kernel_size // 2
        self.cur_target_indices = []

        # Wav2Vec2Frontend
        print('Wav2Vec2Frontend model loading ...')
        # dynamic bmodel
        self.wav2Vec2Frontend_net = sail.Engine(args.encoder_frontend_bmodel, args.dev_id, sail.IOMode.DEVIO)
        logging.info("load {} success!".format(args.encoder_frontend_bmodel))
        self.wav2Vec2Frontend_graph_name = self.wav2Vec2Frontend_net.get_graph_names()[0]
        self.wav2Vec2Frontend_input_names = self.wav2Vec2Frontend_net.get_input_names(self.wav2Vec2Frontend_graph_name)
        self.wav2Vec2Frontend_output_names = self.wav2Vec2Frontend_net.get_output_names(self.wav2Vec2Frontend_graph_name)
        self.wav2Vec2Frontend_output_tensors = {}
        for output_name in self.wav2Vec2Frontend_output_names:
            output_shape = self.wav2Vec2Frontend_net.get_output_shape(self.wav2Vec2Frontend_graph_name, output_name)
            output_dtype = self.wav2Vec2Frontend_net.get_output_dtype(self.wav2Vec2Frontend_graph_name, output_name)
            output = sail.Tensor(self.handle, output_shape, output_dtype, True, True)
            self.wav2Vec2Frontend_output_tensors[output_name] = output
        input_shape = self.wav2Vec2Frontend_net.get_input_shape(self.wav2Vec2Frontend_graph_name, self.wav2Vec2Frontend_input_names[0])
        self.max_input_len = input_shape[1]
        print('Wav2Vec2Frontend model loaded')

        # UnitYEncoderAdaptor
        print('UnitYEncoderAdaptor model loading ...')
        self.unitY_encoder_adaptor_net = sail.Engine(args.encoder_bmodel, args.dev_id, sail.IOMode.DEVIO)
        logging.info("load {} success!".format(args.encoder_bmodel))
        self.unitY_encoder_adaptor_graph_name = self.unitY_encoder_adaptor_net.get_graph_names()[0]
        self.unitY_encoder_adaptor_input_names = self.unitY_encoder_adaptor_net.get_input_names(self.unitY_encoder_adaptor_graph_name)
        self.unitY_encoder_adaptor_output_names = self.unitY_encoder_adaptor_net.get_output_names(self.unitY_encoder_adaptor_graph_name)
        self.unitY_encoder_adaptor_input_shapes = {}
        self.unitY_encoder_adaptor_output_shapes = {}
        self.unitY_encoder_adaptor_output_dtype = {}
        for input_name in self.unitY_encoder_adaptor_input_names:
            self.unitY_encoder_adaptor_input_shapes[input_name] = self.unitY_encoder_adaptor_net.get_input_shape(self.unitY_encoder_adaptor_graph_name, input_name)
        for output_name in self.unitY_encoder_adaptor_output_names:
            self.unitY_encoder_adaptor_output_shapes[output_name] = self.unitY_encoder_adaptor_net.get_output_shape(self.unitY_encoder_adaptor_graph_name, output_name)
            self.unitY_encoder_adaptor_output_dtype[output_name] = self.unitY_encoder_adaptor_net.get_output_dtype(self.unitY_encoder_adaptor_graph_name, output_name)
        print('UnitYEncoderAdaptor model loaded')

        # max input length of unity
        # NOTE: the value effect the final result, the better value is 320
        self.unity_max_encoder_input_length = self.unitY_encoder_adaptor_input_shapes[self.unitY_encoder_adaptor_input_names[0]][1]
        # self.unity_max_encoder_frontend_input_length must be even
        self.unity_max_encoder_frontend_input_length = 2 * self.unity_max_encoder_input_length
        self.unity_max_encoder_output_length = self.unitY_encoder_adaptor_output_shapes[self.unitY_encoder_adaptor_output_names[0]][1]

        self.tokenizer = NllbTokenizer(args.tokenizer_model, ['afr', 'amh', 'arb', 'ary', 'arz', 'asm', 
                                                              'azj', 'bel', 'ben', 'bos', 'bul', 'cat', 'ceb', 
                                                              'ces', 'ckb', 'cmn', 'cmn_Hant', 'cym', 'dan', 
                                                              'deu', 'ell', 'eng', 'est', 'eus', 'fin', 'fra', 
                                                              'fuv', 'gaz', 'gle', 'glg', 'guj', 'heb', 'hin', 
                                                              'hrv', 'hun', 'hye', 'ibo', 'ind', 'isl', 'ita', 
                                                              'jav', 'jpn', 'kan', 'kat', 'kaz', 'khk', 'khm', 'kir', 
                                                              'kor', 'lao', 'lit', 'lug', 'luo', 'lvs', 'mai', 'mal', 
                                                              'mar', 'mkd', 'mlt', 'mni', 'mya', 'nld', 'nno', 'nob', 
                                                              'npi', 'nya', 'ory', 'pan', 'pbt', 'pes', 'pol', 'por', 
                                                              'ron', 'rus', 'sat', 'slk', 'slv', 'sna', 'snd', 'som', 
                                                              'spa', 'srp', 'swe', 'swh', 'tam', 'tel', 'tgk', 'tgl', 
                                                              'tha', 'tur', 'ukr', 'urd', 'uzn', 'vie', 'yor', 'yue', 
                                                              'zsm', 'zul'], 'eng')
        self.eos_idx = self.tokenizer.vocab_info.eos_idx
        self.pad_idx = self.tokenizer.vocab_info.pad_idx
        self.token_encoder = self.tokenizer.create_encoder(lang=self.tgt_lang, mode="target")
        prefix_indices = self.token_encoder.prefix_indices
        assert prefix_indices is not None
        self.prefix_indices: List[int] = prefix_indices.tolist()

        # monotonic max input length = 64
        self.monotonic_max_input_length = 64
        print('monotonic text decoder frontend model loading ...')
        self.monotonic_text_decoder_frontend_net = sail.Engine(args.decoder_frontend_bmodel, args.dev_id, sail.IOMode.DEVIO)
        logging.info("load {} success!".format(args.decoder_frontend_bmodel))
        self.monotonic_text_decoder_frontend_graph_name = self.monotonic_text_decoder_frontend_net.get_graph_names()[0]
        self.monotonic_text_decoder_frontend_input_names = self.monotonic_text_decoder_frontend_net.get_input_names(self.monotonic_text_decoder_frontend_graph_name)
        self.monotonic_text_decoder_frontend_output_names = self.monotonic_text_decoder_frontend_net.get_output_names(self.monotonic_text_decoder_frontend_graph_name)
        self.monotonic_text_decoder_frontend_output_tensors = {}
        for output_name in self.monotonic_text_decoder_frontend_output_names:
            output_shape = self.monotonic_text_decoder_frontend_net.get_output_shape(self.monotonic_text_decoder_frontend_graph_name, output_name)
            output_dtype = self.monotonic_text_decoder_frontend_net.get_output_dtype(self.monotonic_text_decoder_frontend_graph_name, output_name)
            output = sail.Tensor(self.handle, output_shape, output_dtype, True, True)
            self.monotonic_text_decoder_frontend_output_tensors[output_name] = output
        self.monotonic_text_decoder_frontend_input_shapes = {}
        for input_name in self.monotonic_text_decoder_frontend_input_names:
            self.monotonic_text_decoder_frontend_input_shapes[input_name] = self.monotonic_text_decoder_frontend_net.get_input_shape(self.monotonic_text_decoder_frontend_graph_name, input_name)
        print('monotonic text decoder frontend model loaded')

        # monotonic max kvcache length = 4096
        self.monotonic_max_kvcache_length = 64 # 4096
        self.monotonic_layer_num = 24
        print('monotonic text decoder model loading ...')
        self.monotonic_text_decoder_net = sail.Engine(args.decoder_step_bigger_1_bmodel, args.dev_id, sail.IOMode.DEVIO)
        logging.info("load {} success!".format(args.decoder_step_bigger_1_bmodel))
        self.monotonic_text_decoder_graph_name = self.monotonic_text_decoder_net.get_graph_names()[0]
        self.monotonic_text_decoder_net.set_io_mode(self.monotonic_text_decoder_graph_name, sail.IOMode.DEVIO)
        self.monotonic_text_decoder_input_names = self.monotonic_text_decoder_net.get_input_names(self.monotonic_text_decoder_graph_name)
        self.monotonic_text_decoder_output_names = self.monotonic_text_decoder_net.get_output_names(self.monotonic_text_decoder_graph_name)
        self.monotonic_text_decoder_output_tensors = {}
        self.kvcache_tensors = []
        for output_name in self.monotonic_text_decoder_output_names:
            output_shape = self.monotonic_text_decoder_net.get_output_shape(self.monotonic_text_decoder_graph_name, output_name)
            output_dtype = self.monotonic_text_decoder_net.get_output_dtype(self.monotonic_text_decoder_graph_name, output_name)
            output = sail.Tensor(self.handle, output_shape, output_dtype, True, True)
            self.monotonic_text_decoder_output_tensors[output_name] = output
        for layer_idx in range(self.monotonic_layer_num):
            self.kvcache_tensors.append(self.monotonic_text_decoder_output_tensors[self.monotonic_text_decoder_output_names[2 + layer_idx]])
        for layer_idx in range(self.monotonic_layer_num):
            self.kvcache_tensors.append(self.monotonic_text_decoder_output_tensors[self.monotonic_text_decoder_output_names[2 + self.monotonic_layer_num + layer_idx]])
        self.monotonic_text_decoder_input_shapes = {}
        for input_name in self.monotonic_text_decoder_input_names:
            self.monotonic_text_decoder_input_shapes[input_name] = self.monotonic_text_decoder_net.get_input_shape(self.monotonic_text_decoder_graph_name, input_name)
        print('monotonic text decoder model loaded')
        print('monotonic text decoder step=0 model loading ...')
        self.monotonic_text_decoder_step0_net = sail.Engine(args.decoder_step_equal_1_bmodel, args.dev_id, sail.IOMode.DEVIO)
        logging.info("load {} success!".format(args.decoder_step_equal_1_bmodel))
        self.monotonic_text_decoder_step0_graph_name = self.monotonic_text_decoder_step0_net.get_graph_names()[0]
        self.monotonic_text_decoder_step0_input_names = self.monotonic_text_decoder_step0_net.get_input_names(self.monotonic_text_decoder_step0_graph_name)
        self.monotonic_text_decoder_step0_output_names = self.monotonic_text_decoder_step0_net.get_output_names(self.monotonic_text_decoder_step0_graph_name)
        self.monotonic_text_decoder_step0_output_tensors = {}
        for output_name in self.monotonic_text_decoder_step0_output_names:
            output_shape = self.monotonic_text_decoder_step0_net.get_output_shape(self.monotonic_text_decoder_step0_graph_name, output_name)
            output_dtype = self.monotonic_text_decoder_step0_net.get_output_dtype(self.monotonic_text_decoder_step0_graph_name, output_name)
            output = sail.Tensor(self.handle, output_shape, output_dtype, True, True)
            self.monotonic_text_decoder_step0_output_tensors[output_name] = output
        self.monotonic_text_decoder_step0_input_shapes = {}
        for input_name in self.monotonic_text_decoder_step0_input_names:
            self.monotonic_text_decoder_step0_input_shapes[input_name] = self.monotonic_text_decoder_step0_net.get_input_shape(self.monotonic_text_decoder_step0_graph_name, input_name)
        print('monotonic text decoder step=0 model loaded')
        self.p_choose_start_layer = 0
        self.decision_method = 'mean' # NOTE: original implement: "min"

        print('monotonic final proj model loading ...')
        self.monotonic_final_proj_net = sail.Engine(args.decoder_final_proj_bmodel, args.dev_id, sail.IOMode.DEVIO)
        logging.info("load {} success!".format(args.decoder_final_proj_bmodel))
        self.monotonic_final_proj_graph_name = self.monotonic_final_proj_net.get_graph_names()[0]
        self.monotonic_final_proj_input_names = self.monotonic_final_proj_net.get_input_names(self.monotonic_final_proj_graph_name)
        self.monotonic_final_proj_output_names = self.monotonic_final_proj_net.get_output_names(self.monotonic_final_proj_graph_name)
        self.monotonic_final_proj_output_tensors = {}
        for output_name in self.monotonic_final_proj_output_names:
            output_shape = self.monotonic_final_proj_net.get_output_shape(self.monotonic_final_proj_graph_name, output_name)
            output_dtype = self.monotonic_final_proj_net.get_output_dtype(self.monotonic_final_proj_graph_name, output_name)
            output = sail.Tensor(self.handle, output_shape, output_dtype, True, True)
            self.monotonic_final_proj_output_tensors[output_name] = output
        self.monotonic_final_proj_input_shapes = {}
        for input_name in self.monotonic_final_proj_input_names:
            self.monotonic_final_proj_input_shapes[input_name] = self.monotonic_final_proj_net.get_input_shape(self.monotonic_final_proj_graph_name, input_name)
        print('monotonic final proj model loaded')

        self.decision_threshold = 0.5
        self.max_len_a = 0 
        self.max_len_b = 100
        self.max_consecutive_writes = 50
        
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0
        self.output_tokens_num = 0
    
    def init(self):
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0
            
    def len_sample_to_ms(self, length, sample_rate):
        return length * 1000 / sample_rate

    def reset(self):
        self.previous_residual_samples = []
        self.cur_target_indices = []
        self.cur_input = []

    def preprocess(self, ori_audio):
        preprocessed_audio = {}
        if self.use_slience_remover:
            # NOTE: original implement: maybe delete some audio, maybe better
            preprocessed_audio['seqs'] = self.silence_remover(torch.tensor(ori_audio['seqs']).squeeze(), self.is_standardized)
        else:
            # NOTE: following code keep whole audio, maybe better
            preprocessed_audio['seqs'] = torch.tensor(ori_audio['seqs']).squeeze().tolist()

        num_samples = math.ceil(self.chunk_duration_ms / 1000 * ori_audio['samplerate'])
        step = 0
        segments = []
        while step < len(preprocessed_audio['seqs']):
            if step + num_samples >= len(preprocessed_audio['seqs']):
                samples = preprocessed_audio['seqs'][step :]
                is_finished = True
            else:
                samples = preprocessed_audio['seqs'][step : step + num_samples]
                is_finished = False
            step = min(step + num_samples, len(preprocessed_audio['seqs']))

            segment = {'index': self.len_sample_to_ms(step, ori_audio['samplerate']), 'content': samples, 
                       'sample_rate': ori_audio['samplerate'],
                        'finished': is_finished,
                        'tgt_lang': self.tgt_lang}
            segments.append(segment)
        return segments
    
    def online_feature_extractor_agent_predict(self, segments):
        outputs = []
        for segment in segments:
            output = {'finished': segment['finished']}
            segment = segment['content']
            samples = self.previous_residual_samples + segment
            if len(samples) < self.num_samples_per_window:
                self.previous_residual_samples = samples
                continue
            num_frames = math.floor(
                (len(samples) - self.len_ms_to_samples(self.window_size - self.shift_size))
                / self.num_samples_per_shift
            )
            # the number of frames used for feature extraction
            # including some part of the previous segment
            effective_num_samples = int(
                num_frames * self.len_ms_to_samples(self.shift_size)
                + self.len_ms_to_samples(self.window_size - self.shift_size)
            )

            input_samples = samples[:effective_num_samples]
            self.previous_residual_samples = samples[
                num_frames * self.num_samples_per_shift :
            ]

            data: WaveformToFbankInput = {
                "waveform": torch.tensor(input_samples).unsqueeze(0),
                "sample_rate": self.sample_rate,
            }
            output['content'] = self.convert_to_fbank(data)["fbank"]
            output['content'] = output['content'].tolist()
            outputs.append(output)
        return outputs
    
    def wav2Vec2Frontend_predict(self, seqs):
        input_data = {self.wav2Vec2Frontend_input_names[0]: seqs}
        wav2Vec2Frontend_input_shapes = {}
        for input_name in self.wav2Vec2Frontend_input_names:
            wav2Vec2Frontend_input_shapes[input_name] = input_data[input_name].shape()
        self.wav2Vec2Frontend_net.process(self.wav2Vec2Frontend_graph_name, input_data, wav2Vec2Frontend_input_shapes, self.wav2Vec2Frontend_output_tensors)
        output_name = self.wav2Vec2Frontend_output_names[0]
        return self.wav2Vec2Frontend_output_tensors[output_name]
    
    def unitY_encoder_adaptor_predict(self, seqs, seqs_len):
        seqs_len = sail.Tensor(self.handle, np.array([seqs_len], dtype=np.int32), True, True)
        seqs_len.sync_s2d()
        input_data = {self.unitY_encoder_adaptor_input_names[0]: seqs,
                        self.unitY_encoder_adaptor_input_names[1]: seqs_len}
        unitY_encoder_adaptor_output_tensors = {}
        for output_name in self.unitY_encoder_adaptor_output_names:
            output = sail.Tensor(self.handle, self.unitY_encoder_adaptor_output_shapes[output_name], self.unitY_encoder_adaptor_output_dtype[output_name], True, True)
            unitY_encoder_adaptor_output_tensors[output_name] = output
        self.unitY_encoder_adaptor_net.process(self.unitY_encoder_adaptor_graph_name, input_data, self.unitY_encoder_adaptor_input_shapes, unitY_encoder_adaptor_output_tensors)
        output_name = self.unitY_encoder_adaptor_output_names[0]
        return unitY_encoder_adaptor_output_tensors[output_name]
    
    def offline_wav2Vec_bert_encoder_agent_predict(self, inputs):
        outputs = []
        valid_encoder_output_lens = []
        finish = False
        for inp in inputs:
            self.cur_input += inp['content']
            if (
                self.min_starting_wait is not None
                and len(self.cur_input) < self.min_starting_wait
                and not inp['finished']
            ):
                continue
            if len(self.cur_input) < self.min_input_length:
                if inp['finished']:
                    return [], [], True
                else:
                    continue
            # original code
            # inps = torch.stack(cur_input).to(device="cpu", dtype=torch.float32)
            # src = self.collate(inps)
            # seqs, padding_mask = get_seqs_and_padding_mask(src)
            inps = np.stack(self.cur_input).astype(np.float32)
            padding_indx = inps.shape[0]
            padding_indx //= self.stride
            inps = inps[None]
            # static
            """
            inps = sail.Tensor(self.handle, inps, True, True)
            inps.sync_s2d()
            if inps.shape()[1] > self.unity_max_encoder_frontend_input_length:
                raise ValueError('error value: inps.shape[1] > self.unity_max_encoder_frontend_input_length')
            else:
                padding_inps = sail.Tensor(self.handle, np.zeros((inps.shape()[0], self.unity_max_encoder_frontend_input_length, inps.shape()[2]), dtype=np.float32), True, True)
                padding_inps.sync_s2d()
                padding_inps.sync_d2d(inps, 0, 0, inps.shape()[0] * inps.shape()[1] * inps.shape()[2])
                inps = padding_inps
            """
            # dynamic
            if inps.shape[1] > self.unity_max_encoder_frontend_input_length:
                raise ValueError('error value: inps.shape[1] > self.unity_max_encoder_frontend_input_length')
            if inps.shape[1] % 2 != 0:                
                inps = np.concatenate((inps, np.zeros((inps.shape[0], 1, inps.shape[2]), dtype=inps.dtype)), axis=1)
            if self.dev_type == "BM1688":
                seqs_len = inps.shape[1]
                if seqs_len != self.max_input_len:
                    inps = np.concatenate((inps, np.zeros((inps.shape[0], self.max_input_len - seqs_len, inps.shape[2]), dtype=inps.dtype)), axis=1)
            inps = sail.Tensor(self.handle, inps, True, True)
            inps.sync_s2d()
            
            seqs = self.wav2Vec2Frontend_predict(inps)
            # static
            # valid_encoder_output_len = ((padding_indx + 2 * self.adapter_pad - self.adapter_kernel_size) // self.adapter_stride) + 1
            # dynamic
            # valid_encoder_output_len = ((seqs.shape()[1] + 2 * self.adapter_pad - self.adapter_kernel_size) // self.adapter_stride) + 1
            valid_encoder_output_len = ((padding_indx + 2 * self.adapter_pad - self.adapter_kernel_size) // self.adapter_stride) + 1
            seqs_len = seqs.shape()[1]
            if seqs_len > self.unity_max_encoder_input_length:
                raise ValueError('error value: seqs.shape[1] > self.unity_max_encoder_input_length')
            else:
                # static
                # pass
                # dynamic
                padding_inps = sail.Tensor(self.handle, np.zeros((seqs.shape()[0], self.unity_max_encoder_input_length, seqs.shape()[2]), dtype=np.float32), True, True)
                padding_inps.sync_s2d()
                padding_inps.sync_d2d(seqs, 0, 0, seqs.shape()[0] * seqs.shape()[1] * seqs.shape()[2])
                seqs = padding_inps
             
            seqs = self.unitY_encoder_adaptor_predict(seqs, padding_indx)
            outputs.append(seqs)
            finish |= inp['finished']
            valid_encoder_output_lens.append(valid_encoder_output_len)
            # TODO: following implement check precision
            # cur_input = []
        return outputs, valid_encoder_output_lens, finish

    def enforce_tgt_lang_in_prefix(self):
        tgt_lang_tag = f"__{self.tgt_lang}__"
        tgt_lang_tag_idx = self.tokenizer.model.token_to_index(tgt_lang_tag)
        self.prefix_indices[-1] = tgt_lang_tag_idx
    
    def monotonic_text_decoder_frontend_predict(self, cur_step: int, seqs: List[int]):
        """only support 1b"""
        assert len(seqs) <= self.monotonic_max_input_length, "bigger than max input length in monotonic decoder"
        assert cur_step >= 0
        seqs_tensor = sail.Tensor(self.handle, np.array([seqs + [0] * (self.monotonic_max_input_length - len(seqs))]).astype(np.int32), True, True)
        seqs_tensor.sync_s2d()
        if self.dev_type == "BM1688":
            cur_step_tensor = self.pos.freqs[cur_step : cur_step + seqs_tensor.shape()[1]].unsqueeze(0)
            cur_step_tensor = sail.Tensor(self.handle, cur_step_tensor.cpu().numpy(), True, True)
        elif self.dev_type == "BM1684X":
            cur_step_tensor = sail.Tensor(self.handle, np.array([cur_step]).astype(np.int32), True, True)
        cur_step_tensor.sync_s2d()
        input_data = {self.monotonic_text_decoder_frontend_input_names[0]: seqs_tensor, 
                    self.monotonic_text_decoder_frontend_input_names[1]: cur_step_tensor}
        self.monotonic_text_decoder_frontend_net.process(self.monotonic_text_decoder_frontend_graph_name, input_data, self.monotonic_text_decoder_frontend_input_shapes, self.monotonic_text_decoder_frontend_output_tensors)
        output_name = self.monotonic_text_decoder_frontend_output_names[0]
        output_tensor = self.monotonic_text_decoder_frontend_output_tensors[output_name]
        if cur_step > 0:
            new_output_tensor = sail.Tensor(self.handle, (output_tensor.shape()[0], 1, output_tensor.shape()[2]), output_tensor.dtype(), True, True)
            new_output_tensor.sync_d2d(output_tensor, 0, 0, output_tensor.shape()[0] * 1 * output_tensor.shape()[2])
            output_tensor = new_output_tensor
        return output_tensor, len(seqs)
    
    def monotonic_text_decoder_predict(self, cur_step: int, seqs, seqs_valid_len, encoder_output, valid_encoder_output_len, kcache, vcache, valid_kv_len):
        assert cur_step >= 0
        if cur_step > 0:
            assert seqs_valid_len == 1
            self_attn_mask = np.ones((seqs_valid_len, self.monotonic_max_kvcache_length), dtype=np.float32) * (-1000)
            self_attn_mask[:, -valid_kv_len-seqs_valid_len:] = 0
            cross_attn_mask = np.ones((seqs_valid_len, self.unity_max_encoder_output_length), dtype=np.float32) * (-1000)
            cross_attn_mask[:, :valid_encoder_output_len] = 0
            self_attn_mask = sail.Tensor(self.handle, self_attn_mask, True, True)
            cross_attn_mask = sail.Tensor(self.handle, cross_attn_mask, True, True)
            self_attn_mask.sync_s2d()
            cross_attn_mask.sync_s2d()

            input_data = {self.monotonic_text_decoder_input_names[0]: seqs, 
                      self.monotonic_text_decoder_input_names[1]: self_attn_mask,
                      self.monotonic_text_decoder_input_names[2]: encoder_output,
                      self.monotonic_text_decoder_input_names[3]: cross_attn_mask,
                      }

            for layer_idx in range(self.monotonic_layer_num):
                input_data[self.monotonic_text_decoder_input_names[4 + layer_idx]] = kcache[layer_idx]
            for layer_idx in range(self.monotonic_layer_num):
                input_data[self.monotonic_text_decoder_input_names[4 + self.monotonic_layer_num + layer_idx]] = vcache[layer_idx]

            self.monotonic_text_decoder_net.process(self.monotonic_text_decoder_graph_name, input_data, self.monotonic_text_decoder_input_shapes, self.monotonic_text_decoder_output_tensors)
            output_seqs, p_choose = self.monotonic_text_decoder_output_tensors[self.monotonic_text_decoder_output_names[0]], self.monotonic_text_decoder_output_tensors[self.monotonic_text_decoder_output_names[1]]
        else:
            self_attn_mask = np.ones((self.monotonic_max_input_length, self.monotonic_max_input_length), dtype=np.float32) * (-1000)
            self_attn_mask = np.triu(self_attn_mask, 1)
            self_attn_mask[seqs_valid_len:] = -1000
            cross_attn_mask = np.zeros((self.monotonic_max_input_length, self.unity_max_encoder_output_length), dtype=np.float32)
            cross_attn_mask[seqs_valid_len:] = -1000
            cross_attn_mask[:, valid_encoder_output_len:] = -1000
            self_attn_mask = sail.Tensor(self.handle, self_attn_mask, True, True)
            cross_attn_mask = sail.Tensor(self.handle, cross_attn_mask, True, True)
            self_attn_mask.sync_s2d()
            cross_attn_mask.sync_s2d()
            
            # seqs with padding
            input_data = {self.monotonic_text_decoder_step0_input_names[0]: seqs, 
                      self.monotonic_text_decoder_step0_input_names[1]: self_attn_mask,
                      self.monotonic_text_decoder_step0_input_names[2]: encoder_output,
                      self.monotonic_text_decoder_step0_input_names[3]: cross_attn_mask,
                      }
            outputs = self.monotonic_text_decoder_step0_net.process(self.monotonic_text_decoder_step0_graph_name, input_data, self.monotonic_text_decoder_step0_input_shapes, self.monotonic_text_decoder_step0_output_tensors)
            output_seqs, p_choose = self.monotonic_text_decoder_step0_output_tensors[self.monotonic_text_decoder_step0_output_names[0]], self.monotonic_text_decoder_step0_output_tensors[self.monotonic_text_decoder_step0_output_names[1]]
        
        if cur_step > 0:
            for layer_idx in range(self.monotonic_layer_num):
                kcache[layer_idx] = self.monotonic_text_decoder_output_tensors[self.monotonic_text_decoder_output_names[2 + layer_idx]]
            for layer_idx in range(self.monotonic_layer_num):
                vcache[layer_idx] = self.monotonic_text_decoder_output_tensors[self.monotonic_text_decoder_output_names[2 + self.monotonic_layer_num + layer_idx]]
        else:
            for layer_idx in range(self.monotonic_layer_num):
                kcache[layer_idx] = self.monotonic_text_decoder_step0_output_tensors[self.monotonic_text_decoder_step0_output_names[2 + layer_idx]]
            for layer_idx in range(self.monotonic_layer_num):
                vcache[layer_idx] = self.monotonic_text_decoder_step0_output_tensors[self.monotonic_text_decoder_step0_output_names[2 + self.monotonic_layer_num + layer_idx]]
        return output_seqs, p_choose, kcache, vcache

    def monotonic_project_predict(self, seqs):
        input_data = {self.monotonic_final_proj_input_names[0]: seqs}
        self.monotonic_final_proj_net.process(self.monotonic_final_proj_graph_name, input_data, self.monotonic_final_proj_input_shapes, self.monotonic_final_proj_output_tensors)
        return self.monotonic_final_proj_output_tensors[self.monotonic_final_proj_output_names[0]]

    def monotonic_predict(self, cur_step: int, target_indices: List[int], pred_seq_list: List[int], encoder_output, valid_encoder_output_len, kcache, vcache, valid_kv_len):
        # generate monotonic input
        if len(pred_seq_list) == 0:
            assert cur_step == 0
            self.enforce_tgt_lang_in_prefix()
            seqs = self.prefix_indices + target_indices
        else:
            seqs = pred_seq_list[-1:]
        embeds, seqs_valid_len = self.monotonic_text_decoder_frontend_predict(cur_step, seqs)
        output_seqs, p_choose, kcache, vcache = self.monotonic_text_decoder_predict(cur_step, embeds, 
                                                                                    seqs_valid_len, encoder_output, 
                                                                                    valid_encoder_output_len, kcache, 
                                                                                    vcache, valid_kv_len)

        output_seqs.reshape([output_seqs.shape()[1], output_seqs.shape()[2]])
        network_output_seqs = output_seqs
        output_seqs = sail.Tensor(output_seqs, [[seqs_valid_len-1, seqs_valid_len], [0, output_seqs.shape()[1]]], True)
        logits = self.monotonic_project_predict(output_seqs)
        network_output_seqs.reshape([1, network_output_seqs.shape()[0], network_output_seqs.shape()[1]])
        logits.sync_d2s()
        logits = logits.asnumpy()
        index = int(np.argmax(logits, axis=2)[0])

        p_choose.sync_d2s()
        p_choose = p_choose.asnumpy()
        _, tgt_len, src_len = p_choose.shape
        p_choose = p_choose.reshape(self.monotonic_layer_num, -1, tgt_len, src_len)

        # NOTE: following implement is same with original code, due to padding
        # original code: prob = p_choose[self.p_choose_start_layer :, :, seqs_valid_len-1, math.ceil(valid_encoder_output_len/2)-1].min()
        if self.decision_method == "min":
            prob = p_choose[self.p_choose_start_layer :, :, seqs_valid_len-1, math.floor(valid_encoder_output_len/2)-1].min()
        elif self.decision_method == "mean":
            prob = p_choose[self.p_choose_start_layer :, :, seqs_valid_len-1, math.floor(valid_encoder_output_len/2)-1].mean()
        else:
            prob = p_choose[self.p_choose_start_layer :, :, seqs_valid_len-1, math.floor(valid_encoder_output_len/2)-1].median()
            
        return index, prob, output_seqs, kcache, vcache, valid_kv_len+seqs_valid_len, seqs_valid_len

    def max_len(self, seqs) -> int:
        return self.max_len_a * int(seqs.shape()[1]) + self.max_len_b

    def predict(self, preprocessed_audio_list):
        assert len(preprocessed_audio_list) == 1
        target_indices = []
        
        for segment in preprocessed_audio_list[0]:
            s_t = time.time()
            seqs = self.online_feature_extractor_agent_predict([segment])
            logging.info("online fbank extraction cost time(ms): " + str((time.time() - s_t) * 1000.))
            s_t = time.time()
            encoder_outputs, valid_encoder_output_lens, finish = self.offline_wav2Vec_bert_encoder_agent_predict(seqs)
            logging.info("online encode cost time(ms): " + str((time.time() - s_t) * 1000.))
            if len(encoder_outputs) == 0:
                continue
            segment_encoder_output, valid_encoder_output_len = encoder_outputs[0], valid_encoder_output_lens[0]

            pred_seq_list = []
            kcaches = [None] * self.monotonic_layer_num
            vcaches = [None] * self.monotonic_layer_num
            valid_kv_len = 0
            step = 0
            finished = False
            s_t = time.time()
            while True:
                # NOTE: check valid_encoder_output_len, due to padding
                index, prob, _, kcaches, vcaches, valid_kv_len, seqs_valid_len = self.monotonic_predict(step, self.cur_target_indices, pred_seq_list, segment_encoder_output, valid_encoder_output_len, kcaches, vcaches, valid_kv_len)

                # NOTE: adjust pad, diff with original code
                if index == self.pad_idx:
                    if len(target_indices) + len(pred_seq_list) > 0:
                        finished = True
                    break
                    
                if prob < self.decision_threshold or index == self.eos_idx:
                    if prob == 1.0:
                        pred_seq_list = []
                    break
                if (
                    finished
                    or index == self.eos_idx
                    or len(target_indices + pred_seq_list) > self.max_len(segment_encoder_output)
                ):
                    finished = True
                    break

                if prob < self.decision_threshold:
                    break

                if (
                    len(target_indices + pred_seq_list) >= self.max_len(segment_encoder_output)
                    or len(pred_seq_list) >= self.max_consecutive_writes
                ):
                    break

                if step == 0:
                    padding_num = (self.monotonic_max_kvcache_length - valid_kv_len) * kcaches[0].shape()[3]
                    for kcache_id in range(len(kcaches)):
                        kcaches_tensor = self.kvcache_tensors[kcache_id]
                        head_ele_num = kcaches_tensor.shape()[2] * kcaches_tensor.shape()[3]
                        layer_ele_num = kcaches_tensor.shape()[1] * head_ele_num
                        cur_head_ele_num = kcaches[kcache_id].shape()[2] * kcaches[kcache_id].shape()[3]
                        cur_layer_ele_num = kcaches[kcache_id].shape()[1] * cur_head_ele_num
                        for layer_id in range(kcaches[kcache_id].shape()[0]):
                            for head_id in range(kcaches[kcache_id].shape()[1]):
                                kcaches_tensor.sync_d2d(kcaches[kcache_id], layer_id * cur_layer_ele_num + head_id * cur_head_ele_num, layer_id * layer_ele_num + head_id * head_ele_num + padding_num, valid_kv_len * kcaches[kcache_id].shape()[3])
                    for vcache_id in range(len(vcaches)):
                        vcaches_tensor = self.kvcache_tensors[self.monotonic_layer_num + vcache_id]
                        head_ele_num = vcaches_tensor.shape()[2] * vcaches_tensor.shape()[3]
                        layer_ele_num = vcaches_tensor.shape()[1] * head_ele_num
                        cur_head_ele_num = vcaches[vcache_id].shape()[2] * vcaches[vcache_id].shape()[3]
                        cur_layer_ele_num = vcaches[vcache_id].shape()[1] * cur_head_ele_num
                        for layer_id in range(vcaches[vcache_id].shape()[0]):
                            for head_id in range(vcaches[vcache_id].shape()[1]):
                                vcaches_tensor.sync_d2d(vcaches[vcache_id], layer_id * cur_layer_ele_num + head_id * cur_head_ele_num, layer_id * layer_ele_num + head_id * head_ele_num + padding_num, valid_kv_len * vcaches[vcache_id].shape()[3])
                    kcaches, vcaches = self.kvcache_tensors[:self.monotonic_layer_num], self.kvcache_tensors[self.monotonic_layer_num:]

                pred_seq_list.append(index)
                step += seqs_valid_len
            logging.info("online decode cost time(ms): " + str((time.time() - s_t) * 1000.))
            
            if len(pred_seq_list) > 0:
                results = "online segment result: " + "".join([self.tokenizer.model.index_to_token(idx) for idx in pred_seq_list])
                logging.info(results.replace('▁', ' '))
            else:
                logging.info("online segment result:")
            target_indices += pred_seq_list
            self.cur_target_indices += pred_seq_list

        if len(target_indices) > 0:
            rec_res =  "".join([self.tokenizer.model.index_to_token(idx) for idx in target_indices])
            return rec_res.replace('▁', ' '), len(target_indices)
        else:
            return "", 0
    
    def __call__(self, audio_list):
        assert len(audio_list) == 1
        preprocessed_audio_list = []
        s_t = time.time()
        for ori_audio in audio_list:
            start_time = time.time()
            preprocessed_audio = self.preprocess(ori_audio)
            self.preprocess_time += time.time() - start_time
            preprocessed_audio_list.append(preprocessed_audio)
        logging.info("online preprocess cost time(ms): " + str((time.time() - s_t) * 1000.))
            
        all_outputs = ''
        for preprocessed_audio in preprocessed_audio_list:
            for seg_si in range(0, len(preprocessed_audio), self.consecutive_segments_num):
                self.reset()
                if seg_si + self.consecutive_segments_num > len(preprocessed_audio):
                    seg_ei = len(preprocessed_audio)
                else:
                    seg_ei = seg_si + self.consecutive_segments_num
                start_time = time.time()
                outputs, output_tokens_num = self.predict([preprocessed_audio[seg_si:seg_ei]])
                self.output_tokens_num += output_tokens_num
                self.inference_time += time.time() - start_time
                all_outputs += outputs.strip()

        return all_outputs
   
def main(args):
    # check params
    if not os.path.exists(args.input):
        raise FileNotFoundError('{} is not existed.'.format(args.input))
    if not os.path.isdir(args.input):
        raise ValueError("{} is not a directory".format(args.input))
    
    bmodels_path = [args.encoder_frontend_bmodel, args.encoder_bmodel, args.tokenizer_model,
                    args.decoder_frontend_bmodel, 
                    args.decoder_step_bigger_1_bmodel, args.decoder_step_equal_1_bmodel,
                    args.decoder_final_proj_bmodel]
    for bmodel_path in bmodels_path:
        if not os.path.exists(bmodel_path):
            raise FileNotFoundError('{} is not existed.'.format(bmodel_path))
    
    # initialize net
    seamless = seamless_stream_s2tt(args)
    seamless.init()
    
    decode_time = 0.0
    # test local audio
    if os.path.isdir(args.input): 
        audio_list = []
        filename_list = []
        results_list = []
        cn = 0
        for root, dirs, filenames in os.walk(args.input):
            for filename in filenames:
                if os.path.splitext(filename)[-1].lower() not in ['.wav']:
                    continue
                audio_file = os.path.join(root, filename)
                cn += 1
                logging.info("{}, audio_file: {}".format(cn, audio_file))
                # decode
                start_time = time.time()
                # librosa
                # seqs, fs = librosa.load(audio_file, sr = None)
                # ffmpeg
                probe = ffmpeg.probe(audio_file)
                bytes, _ = (ffmpeg.input(audio_file).output('-', format='s16le').overwrite_output().global_args('-loglevel', 'warning').run(capture_stdout=True))
                seqs = np.asarray(buf_to_float(bytes))
                fs = float(probe['streams'][0]['sample_rate'])
                decode_time += time.time() - start_time
                
                audio_input = {"seqs" : seqs, "samplerate" : fs}
                audio_list.append(audio_input)
                filename_list.append(filename)
                if (len(audio_list) == 1 or cn == len(filenames)) and len(audio_list):
                    # predict
                    seamless.reset()
                    results = seamless(audio_list)
                    logging.info("whole result: " + results)
                    results_list.append({'filename' : filename_list[0], 'content' : results.strip()})
                        
                    audio_list.clear()
                    filename_list.clear()

        # save results
        if not os.path.exists('results/'):
            os.makedirs('results/') 
        for result in results_list:
            with open(os.path.join('results/', result['filename'].split('/')[-1].split('.')[0]+'.txt'), 'w') as jf:
                jf.write(result['content'].strip())
        logging.info("result saved in {}".format('results dir'))
        
    # calculate speed  
    logging.info("------------------ Predict Time Info ----------------------")
    token_per_second = seamless.output_tokens_num / (decode_time + seamless.preprocess_time + seamless.inference_time)
    decode_time = decode_time / cn
    preprocess_time = seamless.preprocess_time / cn
    inference_time = seamless.inference_time / cn
    logging.info("decode_time(ms): {:.2f}".format(decode_time * 1000))
    logging.info("preprocess_time(ms): {:.2f}".format(preprocess_time * 1000))
    logging.info("inference_time(ms): {:.2f}".format(inference_time * 1000))
    logging.info("token_per_second: {:.2f}".format(token_per_second))

def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input', type=str, default='../datasets/aishell_S0764', help='path of input')
    parser.add_argument('--tgt_lang', type=str, default='cmn', help='output langauge')
    parser.add_argument('--encoder_frontend_bmodel', type=str, default='../models/BM1684X/seamless_streaming_encoder_frontend_fp16_s2t.bmodel', help='path of Wav2Vec2Frontend bmodel')
    parser.add_argument('--encoder_bmodel', type=str, default='../models/BM1684X/seamless_streaming_encoder_fp16_s2t.bmodel', help='path of UnitYEncoderAdaptor bmodel')
    parser.add_argument('--tokenizer_model', type=str, default='../models/tokenizer.model', help='path of tokenizer model')
    parser.add_argument('--decoder_frontend_bmodel', type=str, default='../models/BM1684X/seamless_streaming_decoder_frontend_fp16_s2t.bmodel', help='path of monotonic text decoder frontend bmodel')
    parser.add_argument('--decoder_step_bigger_1_bmodel', type=str, default='../models/BM1684X/seamless_streaming_decoder_step_bigger_1_fp16_s2t.bmodel', help='path of monotonic text decoder bmodel')
    parser.add_argument('--decoder_step_equal_1_bmodel', type=str, default='../models/BM1684X/seamless_streaming_decoder_step_equal_1_fp16_s2t.bmodel', help='path of monotonic text decoder step=0 bmodel')
    parser.add_argument('--decoder_final_proj_bmodel', type=str, default='../models/BM1684X/seamless_streaming_decoder_final_proj_fp16_s2t.bmodel', help='path of monotonic final proj bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')
    parser.add_argument('--sample_rate', type=int, default=16000, help='audio sample ratio')
    parser.add_argument('--use_slience_remover', action='store_true', default=False, help='whether to use slience remover')
    parser.add_argument('--chunk_duration_ms', type=int, default=1600, help='segment length (ms)')
    parser.add_argument('--consecutive_segments_num', type=int, default=1, help='the processed number of segments once')
    parser.add_argument('--fbank_min_input_length', type=int, default=80, help="the min length of fbank input to encoder")
    parser.add_argument('--fbank_min_starting_wait', type=int, default=48, help="the waitting min length of fbank input to encoder, valid when it > fbank_min_input_length")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done.')