#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#

from typing import Optional, Tuple, final
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)
logging.getLogger().setLevel(logging.INFO)
import time

from fairseq2.data import VocabularyInfo
from fairseq2.models.encoder_decoder import EncoderDecoderModel
from fairseq2.models.sequence import SequenceModelOutput
from customized_fairseq2.incremental_state import IncrementalStateBag
from fairseq2.nn.padding import PaddingMask
from overrides import final as finaloverride
from torch import Tensor
import numpy as np
import torch
import sophon.sail as sail


class KVCache:

    def __init__(self, beam_size: int, max_seq_len: int) -> None:

        self.k = torch.zeros((beam_size, 16, max_seq_len, 64), dtype=torch.float32)
        self.v = torch.zeros((beam_size, 16, max_seq_len, 64), dtype=torch.float32)

        self.seq_len = 0

    def set(self, k, v, seq_len) -> None:
        self.k = k
        self.v = v

        self.seq_len = seq_len

    def reorder(self, new_order: Tensor) -> None:
        self.k = self.k.index_select(0, new_order)
        self.v = self.v.index_select(0, new_order)


@final
class UnitYX2TModel(EncoderDecoderModel):
    model_dim: int

    def __init__(
        self,
        encoder_frontend,
        encoder,
        decoder_frontend,
        decoder,
        final_proj,
        dev_id: int,
        target_vocab_info: VocabularyInfo,
    ) -> None:
        self.handle = sail.Handle(dev_id)
        model_dim = -1
        self.layer_num = 24
        self.seq_len = 0
        self.beam_size = 5
        self.kvcache = {}

        super().__init__(model_dim, target_vocab_info)

        self.encoder_frontend = encoder_frontend
        self.encoder = encoder
        self.decoder_frontend = decoder_frontend
        self.decoder = decoder
        self.final_proj = final_proj
        self.target_vocab_info = target_vocab_info

        self.encoder_frontend_graph_name = self.encoder_frontend.get_graph_names()[0]
        self.encoder_frontend_input_names = self.encoder_frontend.get_input_names(self.encoder_frontend_graph_name)
        self.encoder_frontend_output_names = self.encoder_frontend.get_output_names(self.encoder_frontend_graph_name)
        self.encoder_frontend_output_tensors = {}
        for output_name in self.encoder_frontend_output_names:
            output_shape = self.encoder_frontend.get_output_shape(self.encoder_frontend_graph_name, output_name)
            output_dtype = self.encoder_frontend.get_output_dtype(self.encoder_frontend_graph_name, output_name)
            output = sail.Tensor(self.handle, output_shape, output_dtype, True, True)
            self.encoder_frontend_output_tensors[output_name] = output

        self.max_encoder_input_length = 576
        self.max_decoder_output_length = 73
        self.encoder_graph_name = self.encoder.get_graph_names()[0]
        self.encoder_input_names = self.encoder.get_input_names(self.encoder_graph_name)
        self.encoder_output_names = self.encoder.get_output_names(self.encoder_graph_name)
        self.encoder_input_shapes = {}
        self.encoder_output_shapes = {}
        self.encoder_output_dtype = {}
        for input_name in self.encoder_input_names:
            self.encoder_input_shapes[input_name] = self.encoder.get_input_shape(self.encoder_graph_name, input_name)
        for output_name in self.encoder_output_names:
            self.encoder_output_shapes[output_name] = self.encoder.get_output_shape(self.encoder_graph_name, output_name)
            self.encoder_output_dtype[output_name] = self.encoder.get_output_dtype(self.encoder_graph_name, output_name)
        self.encoder_output_tensors = {}
        for output_name in self.encoder_output_names:
            # output = sail.Tensor(self.handle, (self.beam_size, self.max_decoder_output_length, self.encoder_output_shapes[output_name][2]), self.encoder_output_dtype[output_name], True, True)
            output = sail.Tensor(self.handle, self.encoder_output_shapes[output_name], self.encoder_output_dtype[output_name], True, True)
            self.encoder_output_tensors[output_name] = output
        self.encoder_output_beam_expand = sail.Tensor(self.handle, (self.beam_size, self.max_decoder_output_length, self.encoder_output_shapes[output_name][2]), self.encoder_output_dtype[output_name], True, True)

        self.decoder_frontend_graph_name = self.decoder_frontend.get_graph_names()[0]
        self.decoder_frontend_input_names = self.decoder_frontend.get_input_names(self.decoder_frontend_graph_name)
        self.decoder_frontend_output_names = self.decoder_frontend.get_output_names(self.decoder_frontend_graph_name)
        self.decoder_frontend_output_tensors = {}
        for output_name in self.decoder_frontend_output_names:
            output_shape = self.decoder_frontend.get_output_shape(self.decoder_frontend_graph_name, output_name)
            output_dtype = self.decoder_frontend.get_output_dtype(self.decoder_frontend_graph_name, output_name)
            output_shape[1] = 1
            output = sail.Tensor(self.handle, output_shape, output_dtype, True, True)
            self.decoder_frontend_output_tensors[output_name] = output
        self.decoder_frontend_input_shapes = {}
        self.decoder_frontend_output_shapes = {}
        for input_name in self.decoder_frontend_input_names:
            self.decoder_frontend_input_shapes[input_name] = self.decoder_frontend.get_input_shape(self.decoder_frontend_graph_name, input_name)
        for output_name in self.decoder_frontend_output_names:
            self.decoder_frontend_output_shapes[output_name] = self.decoder_frontend.get_output_shape(self.decoder_frontend_graph_name, output_name)
        
        self.decoder_graph_name = self.decoder.get_graph_names()[0]
        self.decoder.set_io_mode(self.decoder_graph_name, sail.IOMode.DEVIO)
        self.decoder_input_names = self.decoder.get_input_names(self.decoder_graph_name)
        self.decoder_output_names = self.decoder.get_output_names(self.decoder_graph_name)
        self.decoder_output_tensors = {}
        self.kvcache_tensors = []
        for output_name in self.decoder_output_names:
            output_shape = self.decoder.get_output_shape(self.decoder_graph_name, output_name)
            output_dtype = self.decoder.get_output_dtype(self.decoder_graph_name, output_name)
            output = sail.Tensor(self.handle, output_shape, output_dtype, True, True)
            self.decoder_output_tensors[output_name] = output
        for layer_idx in range(self.layer_num):
            self.kvcache_tensors.append(self.decoder_output_tensors[self.decoder_output_names[1 + layer_idx]])
        for layer_idx in range(self.layer_num):
            self.kvcache_tensors.append(self.decoder_output_tensors[self.decoder_output_names[1 + self.layer_num + layer_idx]])
        self.decoder_input_shapes = {}
        for input_name in self.decoder_input_names:
            self.decoder_input_shapes[input_name] = self.decoder.get_input_shape(self.decoder_graph_name, input_name)
        
        # self.final_proj_inps = sail.Tensor(self.handle, np.zeros((self.beam_size, self.decoder_output_tensors[self.decoder_output_names[0]].shape()[1], self.decoder_output_tensors[self.decoder_output_names[0]].shape()[2]), dtype=np.float32), True, True)
        # self.final_proj_inps.sync_s2d()
        self.final_proj_graph_name = self.final_proj.get_graph_names()[0]
        self.final_proj_input_names = self.final_proj.get_input_names(self.final_proj_graph_name)
        self.final_proj_output_names = self.final_proj.get_output_names(self.final_proj_graph_name)
        self.final_proj_output_tensors = {}
        for output_name in self.final_proj_output_names:
            output_shape = self.final_proj.get_output_shape(self.final_proj_graph_name, output_name)
            output_dtype = self.final_proj.get_output_dtype(self.final_proj_graph_name, output_name)
            output = sail.Tensor(self.handle, output_shape, output_dtype, True, True)
            self.final_proj_output_tensors[output_name] = output
        self.final_proj_input_shapes = {}
        for input_name in self.final_proj_input_names:
            self.final_proj_input_shapes[input_name] = self.final_proj.get_input_shape(self.final_proj_graph_name, input_name)

        kv_num = self.layer_num * 2
        for i in range(kv_num):
            self.kvcache[i] = np.zeros((self.beam_size, 16, 32, 64), dtype=np.float32) # KVCache(5, 64)
            self.kvcache[i] = sail.Tensor(self.handle, self.kvcache[i], True, True)
            self.kvcache[i].sync_s2d()

    @finaloverride
    def encode(
        self, seqs: Tensor, padding_mask: Optional[PaddingMask]
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        if seqs.shape[1] % 2 != 0:
            offset = -1
        else:
            offset = 0
        seqs = sail.Tensor(self.handle, seqs.numpy(), True, True)
        seqs.sync_s2d()
        seqs_len = seqs.shape()[1]
        input_seqs = seqs
        input_data = {self.encoder_frontend_input_names[0]: input_seqs}
        encoder_frontend_input_shapes = {}
        for input_name in self.encoder_frontend_input_names:
            encoder_frontend_input_shapes[input_name] = input_data[input_name].shape()
        self.encoder_frontend.process(self.encoder_frontend_graph_name, input_data, encoder_frontend_input_shapes, self.encoder_frontend_output_tensors)
        output_name = self.encoder_frontend_output_names[0]
        input_seqs = self.encoder_frontend_output_tensors[output_name]

        # NOTE: input_seqs.shape() is not real shape, it is with padding
        seqs_len //= 2
        # print('seqs_len: ', seqs_len)
        # input_seqs = np.concatenate([input_seqs, np.zeros((1, 576-input_seqs.shape[1], 1024), dtype=input_seqs.dtype)], axis=1)
        # if (input_seqs.shape()[1] < self.max_encoder_input_length):
        #     padding_inps = sail.Tensor(self.handle, np.zeros((input_seqs.shape()[0], self.max_encoder_input_length, input_seqs.shape()[2]), dtype=np.float32), True, True)
        #     padding_inps.sync_s2d()
        #     padding_inps.sync_d2d(input_seqs, 0, 0, input_seqs.shape()[0] * input_seqs.shape()[1] * input_seqs.shape()[2])
        #     input_seqs = padding_inps
        # TODO: according to padding_mask
        if padding_mask is not None:
            padding_mask = PaddingMask(padding_mask.seq_lens // 2, seqs_len)
            offset = padding_mask.seq_lens[0]
            print('UnitYX2TModel encode padding_mask: ', padding_mask.seq_lens, padding_mask.batch_seq_len)
        else:
            assert offset == 0
            offset = seqs_len - offset
            padding_mask = PaddingMask(torch.tensor([offset], dtype=torch.int32), seqs_len)
        # input_data = {self.encoder_input_names[0]: input_seqs,
        #                 self.encoder_input_names[1]: np.array([offset], dtype=int)}
        # outputs = self.encoder.run(self.encoder_output_names, input_data)
        # outputs = torch.from_numpy(outputs[0])
        # outputs = outputs[:, :((seqs_len + 2 * 4 - 8) // 8) + 1]
        offset = sail.Tensor(self.handle, np.array([offset], dtype=np.int32), True, True)
        offset.sync_s2d()
        input_data = {self.encoder_input_names[0]: input_seqs,
                        self.encoder_input_names[1]: offset}
        self.encoder.process(self.encoder_graph_name, input_data, self.encoder_input_shapes, self.encoder_output_tensors)
        output_name = self.encoder_output_names[0]
        outputs = self.encoder_output_tensors[output_name]
        if padding_mask is not None:
            padding_mask = PaddingMask(((padding_mask.seq_lens + 2 * 4 - 8) // 8) + 1, batch_seq_len=outputs.shape()[1])
        # print(outputs.shape)

        # print('UnitYX2TModel encode padding_mask: ', padding_mask.seq_lens, padding_mask.batch_seq_len)
        # print(seqs[:, 0], seqs[0, -1])
        # print(outputs[:, 0], outputs[0, -1])
        # NOTE: the diff is due to padding mask, many invalid ele is computed, reducing 576 to accuracy length might get the lower diff. but diff 0.* and 0.0000* get the similar final proj diff 
        # assert torch.abs(outputs - seqs).sum() < 1
        # return outputs, padding_mask
        # print('UnitYX2TModel encode seqs.shape: ', seqs.shape)
        # return seqs, padding_mask # type: ignore[no-any-return]
        for i in range(5):
            self.encoder_output_beam_expand.sync_d2d(outputs, 0, i * self.encoder_output_beam_expand.shape()[1] * self.encoder_output_beam_expand.shape()[2], padding_mask.seq_lens[0] * outputs.shape()[2])
        return outputs, padding_mask
    
    def reset(self):
        self.seq_len = 0
        self.kvcache = {}
        kv_num = self.layer_num * 2
        for i in range(kv_num):
            self.kvcache[i] = np.zeros((self.beam_size, 16, 32, 64), dtype=np.float32) # KVCache(5, 64)
            self.kvcache[i] = sail.Tensor(self.handle, self.kvcache[i], True, True)
            self.kvcache[i].sync_s2d()

    @finaloverride
    def decode(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        encoder_output: Tensor,
        encoder_padding_mask: Optional[PaddingMask],
        *,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        if self.seq_len == 0:
            state_bag.step_nr = 0
        logging.info('offline decode current step: ' + str(state_bag.step_nr))
        # print('state_bag.max_num_steps: ', state_bag.max_num_steps)
        # print('state_bag._module_states: ', state_bag._module_states)
        # print(seqs.shape)
        padding_mask = PaddingMask(torch.tensor([seqs.shape[0]], dtype=torch.int32), seqs.shape[0])
        input_seqs = seqs.numpy()
        input_seqs = input_seqs.astype(np.int32)
        start_time = time.time()
        if input_seqs.shape[0] != self.beam_size:
            input_seqs = np.concatenate((input_seqs, np.zeros((self.beam_size-input_seqs.shape[0], 1), dtype=np.int32)), axis=0)
        seqs_tensor = sail.Tensor(self.handle, input_seqs, True, True)
        seqs_tensor.sync_s2d()
        cur_step = sail.Tensor(self.handle, np.array([state_bag.step_nr]).astype(np.int32), True, True)
        cur_step.sync_s2d()
        input_data = {self.decoder_frontend_input_names[0]: seqs_tensor, 
                    self.decoder_frontend_input_names[1]: cur_step}
        self.decoder_frontend.process(self.decoder_frontend_graph_name, input_data, self.decoder_frontend_input_shapes, self.decoder_frontend_output_tensors)
        output_name = self.decoder_frontend_output_names[0]
        output = self.decoder_frontend_output_tensors[output_name]
        # if seqs.shape[0] != 5:
        #     outputs = outputs[:seqs.shape[0]]
        # print(torch.abs(outputs - seqs).sum())
        """
        for i in range(len(list(state_bag._module_states.values()))):
            if i % 2 == 0:
                print('list(state_bag._module_states.values())[i] shape: ', list(state_bag._module_states.values())[i].k.shape, list(state_bag._module_states.values())[i].v.shape, list(state_bag._module_states.values())[i].seq_len)
        print('encoder_output.shape: ', encoder_output.shape)
        if encoder_output.shape[0] > 1:
            print(torch.abs(encoder_output[0] - encoder_output[1]).sum())
            print(torch.abs(encoder_output[1] - encoder_output[2]).sum())
            print(torch.abs(encoder_output[2] - encoder_output[3]).sum())
            print(torch.abs(encoder_output[3] - encoder_output[4]).sum())
        """
        logging.info('offline decode frontend cost time(ms): ' + str((time.time() - start_time) * 1000))
        self.seq_len += seqs.shape[1]

        start_time = time.time()
        self_attn_mask = np.ones((1, 32), dtype=np.float32) * (-1000)
        self_attn_mask[:, -self.seq_len:] = 0
        cross_attn_mask = np.ones((1, self.max_decoder_output_length), dtype=np.float32) * (-1000)
        cross_attn_mask[:, :encoder_padding_mask.seq_lens[0]] = 0
        # print(encoder_padding_mask.seq_lens)

        """
        input_data = {self.decoder_input_names[0]: outputs.numpy(), 
                    self.decoder_input_names[1]: self_attn_mask,
                    self.decoder_input_names[2]: encoder_output,
                    self.decoder_input_names[3]: cross_attn_mask,
                    }
        for layer_idx in range(self.layer_num):
            input_data[self.decoder_input_names[4 + layer_idx]] = self.kvcache[layer_idx].k.numpy() # [:, :, :self.seq_len-1]
            input_data[self.decoder_input_names[4 + self.layer_num + layer_idx]] = self.kvcache[layer_idx].v.numpy() # [:, :, :self.seq_len-1]
            # if self.seq_len - 1 != 64:
            #     input_data[self.decoder_input_names[4 + layer_idx]] = np.concatenate((np.zeros((5, 16, 64-self.seq_len+1, 64), dtype=input_data[self.decoder_input_names[4 + layer_idx]].dtype), input_data[self.decoder_input_names[4 + layer_idx]]), axis=2)
            #     input_data[self.decoder_input_names[4 + self.layer_num + layer_idx]] = np.concatenate((np.zeros((5, 16, 64-self.seq_len+1, 64), dtype=input_data[self.decoder_input_names[4 + self.layer_num + layer_idx]].dtype), input_data[self.decoder_input_names[4 + self.layer_num + layer_idx]]), axis=2)
        outputs = self.decoder.run(self.decoder_output_names, input_data)
        """
        self_attn_mask = sail.Tensor(self.handle, self_attn_mask, True, True)
        cross_attn_mask = sail.Tensor(self.handle, cross_attn_mask, True, True)
        self_attn_mask.sync_s2d()
        cross_attn_mask.sync_s2d()

        input_data = {self.decoder_input_names[0]: output, 
                    self.decoder_input_names[1]: self_attn_mask,
                    self.decoder_input_names[2]: self.encoder_output_beam_expand, # self.encoder_output_tensors[self.encoder_output_names[0]],
                    self.decoder_input_names[3]: cross_attn_mask,
                    }

        for layer_idx in range(self.layer_num):
            input_data[self.decoder_input_names[4 + layer_idx]] = self.kvcache[layer_idx]
        for layer_idx in range(self.layer_num):
            input_data[self.decoder_input_names[4 + self.layer_num + layer_idx]] = self.kvcache[self.layer_num + layer_idx]

        # write_input_data = {}
        # for k, v in input_data.items():
        #     v.sync_d2s()
        #     write_input_data[k] = v.asnumpy()
        # np.savez("test.npz", **write_input_data)
        self.decoder.process(self.decoder_graph_name, input_data, self.decoder_input_shapes, self.decoder_output_tensors)
        output_seqs = self.decoder_output_tensors[self.decoder_output_names[0]]
        logging.info('offline decode cost time(ms): ' + str((time.time() - start_time) * 1000))

        for i in range(1, len(self.decoder_output_names)):
            self.kvcache[i-1] = self.decoder_output_tensors[self.decoder_output_names[i]]
        state_bag._module_states = self.kvcache
        # print(torch.abs(torch.from_numpy(outputs[0][:seqs.shape[0]]) - seqs).sum())
        # print(seqs.shape)
        # print(list(state_bag._module_states.values())[0].seq_len)
        # print(outputs[0][:seqs.shape[0]], seqs)
        # return seqs, padding_mask
        return output_seqs, padding_mask

    @finaloverride
    def project(
        self, decoder_output: Tensor, decoder_padding_mask: Optional[PaddingMask]
    ) -> SequenceModelOutput:
        # print(decoder_output.shape)
        # if decoder_output.shape()[0] != 5:
        #     # decoder_output_onnx = decoder_output.expand(5, decoder_output.shape[1], decoder_output.shape[2]).numpy()
        #     self.final_proj_inps.sync_d2d(decoder_output, 0, 0, decoder_output.shape()[0] * decoder_output.shape()[1] * decoder_output.shape()[2])
        #     decoder_output = self.final_proj_inps
        input_data = {self.final_proj_input_names[0]: decoder_output}
        self.final_proj.process(self.final_proj_graph_name, input_data, self.final_proj_input_shapes, self.final_proj_output_tensors)
        output = self.final_proj_output_tensors[self.final_proj_output_names[0]]
        output.sync_d2s()
        # logits = self.final_proj(decoder_output)
        # print(torch.abs(torch.from_numpy(outputs[0][:logits.shape[0]])-logits).sum())
        # print(torch.from_numpy(outputs[0][:logits.shape[0]])-logits)

        return SequenceModelOutput(torch.from_numpy(output.asnumpy())[:decoder_padding_mask.seq_lens[0]], self.target_vocab_info)