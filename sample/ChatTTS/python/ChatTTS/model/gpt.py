import os, platform
from dataclasses import dataclass
import logging
from typing import Union, List, Optional, Tuple, Callable
from pathlib import Path
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P
from numba import jit
import numpy as np
from einops import rearrange
from tqdm import tqdm
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import is_flash_attn_2_available
from ..utils import del_all
import sophon.sail as sail
import time
sail.set_loglevel(sail.LogLevel.ERROR)
class GPT(nn.Module):
    def __init__(
        self,
        model_path: str,
        gpt_config: dict,
        tpu_id: int = 0,
        logger=logging.getLogger(__name__),
    ):
        super().__init__()

        self.logger = logger

        self.dev_ids = [tpu_id]
        self.dev_ids_num = len(self.dev_ids)
        self.handles = {dev: sail.Handle(dev) for dev in self.dev_ids}
        self.handle = self.handles[0]
        self.target = sail.Handle(self.dev_ids[0]).get_target()

        self.model = sail.EngineLLM(model_path, self.dev_ids)
        self.tensors = {}
        self.graph_names = self.model.get_graph_names()
        self.io_alone = 0

        if self.target in ["BM1688", "CV186AH"]:
            for net in self.graph_names:
                self.tensors[net] = {}
                self.tensors[net]["addr_mode"] = self.model.get_addr_mode(net)
                if self.tensors[net]["addr_mode"] == 0:
                    self.tensors[net]['input'] = self.model.create_max_input_tensors(net)
                    self.tensors[net]['output'] = self.model.create_max_output_tensors(net)
                elif self.tensors[net]["addr_mode"] == 1:
                    self.io_alone = 1
                    self.tensors[net]['input'] = self.model.get_input_tensors(net)
                    self.tensors[net]['output'] = self.model.get_output_tensors(net)
        else:
            for net in self.graph_names:
                self.tensors[net] = {}
                self.tensors[net]["addr_mode"] = self.model.get_addr_mode(net)
                if self.tensors[net]["addr_mode"] == 0:
                    self.tensors[net]['input'] = self.model.get_input_tensors_addrmode0(net)
                    self.tensors[net]['output'] = self.model.get_output_tensors_addrmode0(net)
                elif self.tensors[net]["addr_mode"] == 1:
                    self.io_alone = 1
                    self.tensors[net]['input'] = self.model.get_input_tensors(net)
                    self.tensors[net]['output'] = self.model.get_output_tensors(net)

        # initialize params
        self.is_dynamic = self.model.get_is_dynamic("block_0")
        # self.is_dynamic = False
        print("dynamic: ", self.is_dynamic)
        self.text_token_length = 0
        self.code_token_length = 0
        _, self.SEQLEN, self.HIDDEN_SIZE = self.tensors["block_0"]["input"][0].shape()
        _, _, self.ATTEN_HEAD, self.ATTEN_DIM = self.tensors["block_cache_0"]["input"][3].shape()
        self.top_p = 0.7
        self.top_k = 20
        self.temperature = 0.7
        self.repeat_penalty = 1.0
        self.ATTENTION_MASK = -10000.0
        if self.tensors["block_0"]["input"][2].dtype() == sail.Dtype.BM_BFLOAT16:
            self.ATTENTION_MASK = 50716
        self.NUM_LAYERS = 20                    # 20 blocks
        self.config = gpt_config
        self.num_vq = int(gpt_config["num_vq"])
        self.num_audio_tokens = int(gpt_config["num_audio_tokens"])
        self.num_text_tokens = int(gpt_config["num_text_tokens"])
        
        # initialize net name
        self.name_embed_text = "embedding_text"
        self.name_embed_cache_text = "embedding_text_cache"
        # self.name_embed_code = "embedding_code"
        self.name_embed_cache_code = "embedding_code_cache"
        self.name_blocks = ["block_"+str(i) for i in range(self.NUM_LAYERS)]
        self.name_blocks_cache = ["block_cache_"+str(i) for i in range(self.NUM_LAYERS)]
        self.name_lm_text = "lm_head_text"
        self.name_lm_code = "lm_head_code"
        self.greedy_text = "greedy_head_text"
        self.greedy_code = "greedy_head_code"

        self.past_k = {}
        self.past_v = {}
        # not io_alone 
        if self.io_alone == 0 or self.is_dynamic:
            print("no io_alone")
            for j in range(self.NUM_LAYERS):
                self.past_k[j] = {}
                self.past_v[j] = {}
                for i in range(self.dev_ids_num):
                    self.past_k[j][i] = self.init_tensor(self.dev_ids[i], self.tensors[self.name_blocks_cache[j]]["input"][5 * i + 3])
                    self.past_v[j][i] = self.init_tensor(self.dev_ids[i], self.tensors[self.name_blocks_cache[j]]["input"][5 * i + 4])
        else:
            for j in range(self.NUM_LAYERS):
                self.past_k[j] = {}
                self.past_v[j] = {}
                for i in range(self.dev_ids_num):
                    self.past_k[j][i] = self.tensors[self.name_blocks_cache[j]]["input"][5 * i + 3]
                    self.past_v[j][i] = self.tensors[self.name_blocks_cache[j]]["input"][5 * i + 4]
        self.past_kv_cache_tensors={}
        for i in range(self.NUM_LAYERS):
            self.past_kv_cache_tensors[i]={}
            for j in range(self.dev_ids_num):
                self.past_kv_cache_tensors[i][5 * j + 0]={}
                self.past_kv_cache_tensors[i][5 * j + 1]={}
                for k in range(1, self.SEQLEN+1):
                    self.past_kv_cache_tensors[i][5 * j + 0][k] = sail.Tensor(self.past_k[i][j], [1, 1, self.ATTEN_HEAD, self.ATTEN_DIM], (k-1) * (self.ATTEN_HEAD * self.ATTEN_DIM))
                    self.past_kv_cache_tensors[i][5 * j + 1][k] = sail.Tensor(self.past_v[i][j], [1, 1, self.ATTEN_HEAD, self.ATTEN_DIM], (k-1) * (self.ATTEN_HEAD * self.ATTEN_DIM))

        self.first_embed_input_text = self.model.create_max_input_tensors(self.name_embed_text)
        self.first_hidden_state_text = self.model.create_max_output_tensors(self.name_embed_text)
        self.next_embed_input_text = self.model.create_max_input_tensors(self.name_embed_cache_text)
        self.next_hidden_state_text = self.model.create_max_output_tensors(self.name_embed_cache_text)

        self.next_embed_input_code = self.model.create_max_input_tensors(self.name_embed_cache_code)
        self.next_hidden_state_code = self.model.create_max_output_tensors(self.name_embed_cache_code)
        
        self.first_pid = {}
        self.next_pid = {}
        self.first_attention_mask = {}
        self.next_attention_mask = {}
        
        self.lm_text_input = self.model.create_max_input_tensors(self.name_lm_text)
        self.lm_text_output = self.model.create_max_output_tensors(self.name_lm_text)
        
        self.lm_code_input = self.model.create_max_input_tensors(self.name_lm_code)
        self.lm_code_output = self.model.create_max_output_tensors(self.name_lm_code)
        
        for i in range(self.dev_ids_num):
            self.first_pid[i] = self.init_tensor(self.dev_ids[i], self.tensors[self.name_blocks[0]]["input"][1])
            self.first_attention_mask[i] = self.init_tensor(self.dev_ids[i], self.tensors[self.name_blocks[0]]["input"][2])
            self.next_pid[i] = self.init_tensor(self.dev_ids[i], self.tensors[self.name_blocks_cache[0]]["input"][1])
            self.next_attention_mask[i] = self.init_tensor(self.dev_ids[i], self.tensors[self.name_blocks_cache[0]]["input"][2])
        
    def init_input_tensor(self, dev_id, net, index):
        shape = self.model.get_input_shape(net, index)
        type = self.model.get_input_dtype(net, index)
        return sail.Tensor(self.handles[dev_id], shape, type, False, True) 
    
    def init_output_tensor(self, dev_id, net, index):
        shape = self.model.get_output_shape(net, index)
        type = self.model.get_output_dtype(net, index)
        return sail.Tensor(self.handles[dev_id], shape, type, False, True)
    
    def init_tensor(self, dev_id, shape, type):
        return sail.Tensor(self.handles[dev_id], shape, type, False, True) 
    
    def init_tensor(self, dev_id, tensor):
        return sail.Tensor(self.handles[dev_id], tensor.shape(), tensor.dtype(), False, True) 
    
    def type_convert(self, sail_dtype):
        if sail_dtype == sail.Dtype.BM_FLOAT32:
            return np.float32
        if sail_dtype == sail.Dtype.BM_FLOAT16:
            return np.float16
        if sail_dtype == sail.Dtype.BM_INT32:
            return np.int32
        if sail_dtype == sail.Dtype.BM_BFLOAT16: 
            return np.uint16
    
    def get_first_input(self, length, token, infer_text: bool):
        name_embed = self.name_embed_text
        token_length = self.text_token_length if infer_text else self.code_token_length
        input_ids = np.zeros(length, self.type_convert(self.tensors[name_embed]["input"][0].dtype()))
        input_ids[:len(token)] = token

        position_dtype = self.type_convert(self.tensors[self.name_blocks[0]]["input"][1].dtype())
        attention_dtype = self.type_convert(self.tensors[self.name_blocks[0]]["input"][2].dtype())

        position_id = np.arange(length, dtype=position_dtype)
        if token_length < length:
            position_id[token_length:] = 0

        attention_mask = np.full((length, length), self.ATTENTION_MASK, dtype=attention_dtype)
        attention_mask[np.tril_indices(token_length)] = 0

        attention_mask = attention_mask.ravel()

        return input_ids, position_id, attention_mask

    class Context:
        def __init__(self):
            self._interrupt = False

        def set(self, v: bool):
            self._interrupt = v

        def get(self) -> bool:
            return self._interrupt

    @dataclass(repr=False, eq=False)
    class GenerationOutputs:
        ids: List[torch.Tensor]
        attentions: List[Optional[Tuple[torch.FloatTensor, ...]]]
        hiddens: List[torch.Tensor]

        def destroy(self):
            del_all(self.ids)
            del_all(self.attentions)
            del_all(self.hiddens)

    @torch.no_grad()
    def _prepare_generation_outputs(
        self,
        inputs_ids: torch.Tensor,
        start_idx: int,
        end_idx: torch.Tensor,
        attentions: List[Optional[Tuple[torch.FloatTensor, ...]]],
        hiddens: List[torch.Tensor],
        infer_text: bool,
    ) -> GenerationOutputs:
        inputs_ids = [
            inputs_ids[idx].narrow(0, start_idx, i) for idx, i in enumerate(end_idx)
        ]
        if infer_text:
            inputs_ids = [i.narrow(1, 0, 1).squeeze_(1) for i in inputs_ids]

        if len(hiddens) > 0:
            hiddens = torch.stack(hiddens, 1)
            hiddens = [
                hiddens[idx].narrow(0, 0, i) for idx, i in enumerate(end_idx.int())
            ]

        return self.GenerationOutputs(
            ids=inputs_ids,
            attentions=attentions,
            hiddens=hiddens,
        )
        
    def prepare(self, compile=False):
        if self.use_flash_attn and is_flash_attn_2_available():
            self.gpt = self.gpt.to(dtype=torch.float16)
        if compile and not self.is_te_llama and not self.is_vllm:
            try:
                self.compile(backend="inductor", dynamic=True)
                self.gpt.compile(backend="inductor", dynamic=True)
            except RuntimeError as e:
                self.logger.warning(f"compile failed: {e}. fallback to normal mode.")

    def __call__(
        self, input_ids: torch.Tensor, text_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        get_emb
        """
        return super().__call__(input_ids, text_mask)
    def generate_text(
        self, 
        inputs_ids,
        temperature, 
        eos_token, 
    ):
        inputs_ids_list = inputs_ids[0].tolist()
        if isinstance(eos_token, torch.Tensor):
            eos_token = eos_token.item()
        if isinstance(temperature, torch.Tensor):
            temperature = temperature.item()

        # generate
        if len(inputs_ids_list) == 0:
            print("Your text is empty!")
            return []
        token = self.forward_first_text(inputs_ids_list)
        result_tokens = []
        while(token != eos_token and self.text_token_length < self.SEQLEN):
            result_tokens.append(token)
            token = self.forward_next_text()
            
        result_tokens = torch.tensor(result_tokens, dtype=torch.int64).unsqueeze(0).unsqueeze(0)
        return result_tokens
    
    def forward_first_text(self, tokens):
        self.text_token_length = len(tokens)

        length = self.text_token_length + 1 if self.is_dynamic else self.SEQLEN
        # length = self.SEQLEN
        input_ids, position_id, attention_mask = self.get_first_input(length, tokens, True)

        for i in range(self.dev_ids_num):
            # breakpoint()
            self.tensors[self.name_embed_text]["input"][i] = sail.Tensor(self.first_embed_input_text[i], [1, length], 0)
            self.tensors[self.name_embed_text]["output"][i] = sail.Tensor(self.first_hidden_state_text[i], [1, length, self.HIDDEN_SIZE], 0)
            self.tensors[self.name_embed_text]["input"][i].update_data(input_ids.reshape(self.tensors[self.name_embed_text]["input"][i].shape()))
        self.model.process(self.name_embed_text, self.tensors[self.name_embed_text]["input"], self.tensors[self.name_embed_text]["output"])

 
        # blocks
        for i in range(self.dev_ids_num):
            self.tensors[self.name_blocks[0]]["input"][3 * i + 1] = sail.Tensor(self.first_pid[i], [1, length], 0)
            self.tensors[self.name_blocks[0]]["input"][3 * i + 2] = sail.Tensor(self.first_attention_mask[i], [1, 1, length, length], 0)
            self.tensors[self.name_blocks[0]]["input"][3 * i + 1].update_data(position_id.reshape(self.tensors[self.name_blocks[0]]["input"][3 * i + 1].shape()))
            self.tensors[self.name_blocks[0]]["input"][3 * i + 2].update_data(attention_mask.reshape(self.tensors[self.name_blocks[0]]["input"][3 * i + 2].shape()).view(np.uint16))
        for i in range(self.NUM_LAYERS):
            for j in range(self.dev_ids_num):
                self.tensors[self.name_blocks[i]]["input"][3 * j] = sail.Tensor(self.first_hidden_state_text[j], [1, length, self.HIDDEN_SIZE], 0)
                self.tensors[self.name_blocks[i]]["output"][3 * j] = sail.Tensor(self.first_hidden_state_text[j], [1, length, self.HIDDEN_SIZE], 0)
                self.tensors[self.name_blocks[i]]["output"][3 * j + 1] = sail.Tensor(self.past_k[i][j], [1, length, self.ATTEN_HEAD, self.ATTEN_DIM], 0)
                self.tensors[self.name_blocks[i]]["output"][3 * j + 2] = sail.Tensor(self.past_v[i][j], [1, length, self.ATTEN_HEAD, self.ATTEN_DIM], 0)
            if i > 0:
                for j in range(self.dev_ids_num):
                    self.tensors[self.name_blocks[i]]["input"][3 * j + 1] = self.tensors[self.name_blocks[0]]["input"][3 * j + 1]
                    self.tensors[self.name_blocks[i]]["input"][3 * j + 2] = self.tensors[self.name_blocks[0]]["input"][3 * j + 2]
            # breakpoint()
            self.model.process(self.name_blocks[i], self.tensors[self.name_blocks[i]]["input"], self.tensors[self.name_blocks[i]]["output"])

        # breakpoint()
        # lm_head
        self.tensors[self.name_lm_text]["input"][0] = sail.Tensor(self.first_hidden_state_text[0], [1, 1, self.HIDDEN_SIZE], (self.text_token_length - 1) * self.HIDDEN_SIZE)
        self.tensors[self.name_lm_text]["output"][0] = self.lm_text_output[0]
        
        self.model.process(self.name_lm_text, self.tensors[self.name_lm_text]["input"], self.tensors[self.name_lm_text]["output"])

        # sample
        self.tensors[self.greedy_text]["input"][0] = self.tensors[self.name_lm_text]["output"][0]
        self.model.process(self.greedy_text, self.tensors[self.greedy_text]["input"], self.tensors[self.greedy_text]["output"])

        return int(self.tensors[self.greedy_text]["output"][0].asnumpy())

    def forward_next_text(self):
        self.text_token_length += 1
        position_id = np.array(self.text_token_length - 1, self.type_convert(self.tensors[self.name_blocks_cache[0]]["input"][1].dtype()))
        attention_mask = np.zeros(self.SEQLEN+1, self.type_convert(self.tensors[self.name_blocks_cache[0]]["input"][2].dtype()))
        for i in range(self.text_token_length - 1, self.SEQLEN):
            attention_mask[i] = self.ATTENTION_MASK

        # embedding_cache
        if self.dev_ids_num > 1:
            # breakpoint()
            input_ids = np.array(int(self.tensors[self.greedy_text]["output"][0].asnumpy()), self.type_convert(self.tensors[self.name_embed_cache_text]["input"][0].dtype()))
            for i in range(self.dev_ids_num):
                self.next_embed_input_text[i].update_data(input_ids.reshape(self.tensors[self.name_embed_cache_text]["input"][i].shape()))
                self.tensors[self.name_embed_cache_text]["input"][i] = self.next_embed_input_text[i]
        else:
            self.tensors[self.name_embed_cache_text]["input"][0] = self.tensors[self.greedy_text]["output"][0]
        for i in range(self.dev_ids_num):
            self.tensors[self.name_embed_cache_text]["output"][i] = self.next_hidden_state_text[i] 

        self.model.process(self.name_embed_cache_text, self.tensors[self.name_embed_cache_text]["input"], self.tensors[self.name_embed_cache_text]["output"])

        # block_cache
        for i in range(self.dev_ids_num):
            self.tensors[self.name_blocks_cache[0]]["input"][5 * i + 1] = self.next_pid[i]
            self.tensors[self.name_blocks_cache[0]]["input"][5 * i + 2] = self.next_attention_mask[i]
            self.tensors[self.name_blocks_cache[0]]["input"][5 * i + 1].update_data(position_id.reshape(self.tensors[self.name_blocks_cache[0]]["input"][5 * i + 1].shape()))
            self.tensors[self.name_blocks_cache[0]]["input"][5 * i + 2].update_data(attention_mask.reshape(self.tensors[self.name_blocks_cache[0]]["input"][5 * i + 2].shape()).view(np.uint16))


        for i in range(self.NUM_LAYERS):
            for j in range(self.dev_ids_num):
                # breakpoint()
                self.tensors[self.name_blocks_cache[i]]["input"][5 * j] = self.next_hidden_state_text[j]
                self.tensors[self.name_blocks_cache[i]]["output"][3 * j] = self.next_hidden_state_text[j]
                self.tensors[self.name_blocks_cache[i]]["input"][5 * j + 3] = self.past_k[i][j]
                self.tensors[self.name_blocks_cache[i]]["input"][5 * j + 4] = self.past_v[i][j]
                self.tensors[self.name_blocks_cache[i]]["output"][3 * j + 1] = self.past_kv_cache_tensors[i][j * 5 + 0][self.text_token_length]
                self.tensors[self.name_blocks_cache[i]]["output"][3 * j + 2] = self.past_kv_cache_tensors[i][j * 5 + 1][self.text_token_length]
            if i > 0:
                for j in range(self.dev_ids_num):
                    self.tensors[self.name_blocks_cache[i]]["input"][5 * j + 1] = self.tensors[self.name_blocks_cache[0]]["input"][5 * j + 1]
                    self.tensors[self.name_blocks_cache[i]]["input"][5 * j + 2] = self.tensors[self.name_blocks_cache[0]]["input"][5 * j + 2]
            # breakpoint()
            self.model.process(self.name_blocks_cache[i], self.tensors[self.name_blocks_cache[i]]["input"], self.tensors[self.name_blocks_cache[i]]["output"])

        
        #lm_head
        self.tensors[self.name_lm_text]["input"][0] = self.next_hidden_state_text[0]
        # breakpoint()
        self.tensors[self.name_lm_text]["output"][0] = self.lm_text_output[0]
        self.model.process(self.name_lm_text, self.tensors[self.name_lm_text]["input"], self.tensors[self.name_lm_text]["output"])

        # sample
        self.tensors[self.greedy_text]["input"][0] = self.tensors[self.name_lm_text]["output"][0]
        self.model.process(self.greedy_text, self.tensors[self.greedy_text]["input"], self.tensors[self.greedy_text]["output"])

        return int(self.tensors[self.greedy_text]["output"][0].asnumpy())

    def get_speaker(self, inputs_ids, spk_emb):
        temp = torch.where(inputs_ids[0] == 21143)
        if temp[0].shape[0] == 0:
            spk_idx = -1
            spk_emb = list(range(768))
            self.logger.info("Not set speaker")
        else:
            spk_idx = temp[0].item()
            spk_emb = spk_emb.tolist()
        return spk_idx, spk_emb
    
    @torch.no_grad()
    def generate(
        self,
        inputs_ids: torch.Tensor,
        temperature: torch.Tensor,
        eos_token: Union[int, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        max_new_token=2048,
        min_new_token=0,
        logits_processors: Tuple[
            Callable[[torch.LongTensor, torch.FloatTensor], torch.FloatTensor]
        ] = (),
        infer_text=False,
        spk_emb=None,
        return_attn=False,
        return_hidden=False,
        stream=False,
        show_tqdm=True,
        ensure_non_empty=True,
        stream_batch=24,
        manual_seed: Optional[int] = None,
        context=Context(),
    ):
        if infer_text:
            # 文本生成不支持stream输出
            inputs_ids = self.generate_text(
                inputs_ids,
                temperature,
                eos_token,
            )
            yield self.GenerationOutputs(
                ids=[inputs_ids],
                attentions=[],
                hiddens=[],
            )
        else: # audio code generation 支持stream输出
            temperature = temperature.unsqueeze(1)
            attentions = []
            hiddens = []
            stream_iter = 0

            start_idx, end_idx = inputs_ids.shape[1], torch.zeros(inputs_ids.shape[0], device=inputs_ids.device, dtype=torch.long)
            finish = torch.zeros(inputs_ids.shape[0], device=inputs_ids.device).bool()

            pbar = tqdm(total=max_new_token, desc="code", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}(max) [{elapsed}, {rate_fmt}{postfix}]") if show_tqdm else None

            for i in range(max_new_token):
                if i == 0:        
                    logits, hidden = self.forward_first_code_core(inputs_ids[0].tolist(), *self.get_speaker(inputs_ids, spk_emb))
                    inputs_ids = inputs_ids.unsqueeze(2).expand(-1, -1, self.num_vq)
                else:
                    if self.code_token_length == self.SEQLEN:
                        break
                    logits, hidden = self.forward_next_code_core(curr_input_id)
                
                hiddens.append(torch.tensor(hidden, dtype=torch.float32).unsqueeze(0))
                logits = torch.tensor(logits).reshape(self.num_audio_tokens, self.num_vq).transpose(0, 1)
                
                # 应用logits处理器
                inputs_ids_sliced = inputs_ids.narrow(
                    1,
                    start_idx,
                    inputs_ids.size(1) - start_idx,
                ).permute(0, 2, 1)
                logits_token = inputs_ids_sliced.reshape(
                    inputs_ids_sliced.size(0) * inputs_ids_sliced.size(1),
                    -1,
                )
                del inputs_ids_sliced

                for processor in logits_processors:
                    logits = processor(logits_token, logits)
                
                if i < min_new_token:
                    logits[:, eos_token] = -float('inf')
                
                scores = F.softmax(logits / temperature, dim=-1)
                idx_next = torch.multinomial(scores, num_samples=1)
                idx_next = idx_next.view(-1, self.num_vq)
                finish |= (idx_next == eos_token).any(1)
                inputs_ids = torch.cat([inputs_ids, idx_next.unsqueeze(1)], 1)
                curr_input_id = inputs_ids[0, -1].int().tolist()
                
                not_finished = ~finish
                end_idx += not_finished.int()
                stream_iter += not_finished.any().int()

                if stream and stream_iter > 0 and stream_iter % stream_batch == 0:
                    yield self._prepare_generation_outputs(
                        inputs_ids,
                        start_idx,
                        end_idx,
                        attentions,
                        hiddens,
                        infer_text,
                    )

                if finish.all() or context.get():
                    break

                if pbar:
                    pbar.update(1)

            if pbar:
                pbar.close()

            if not finish.all():
                if context.get():
                    self.logger.warning("generation is interrupted")
                else:
                    self.logger.warning(f"incomplete result. Seq_length too small or hit max_new_token: {max_new_token}")

            yield self._prepare_generation_outputs(
                inputs_ids,
                start_idx,
                end_idx,
                attentions,
                hiddens,
                infer_text,
            )
            
    def forward_first_code_core(self, tokens, spk_idx, spk_emb):
        self.code_token_length = len(tokens)

        length = self.text_token_length + 1 if self.is_dynamic else self.SEQLEN
        input_ids, position_id, attention_mask = self.get_first_input(length, tokens, False)

        for i in range(self.dev_ids_num):
            # breakpoint()
            self.tensors[self.name_embed_text]["input"][i] = sail.Tensor(self.first_embed_input_text[i], [1, length], 0)
            self.tensors[self.name_embed_text]["output"][i] = sail.Tensor(self.first_hidden_state_text[i], [1, length, self.HIDDEN_SIZE], 0)
            self.tensors[self.name_embed_text]["input"][i].update_data(input_ids.reshape(self.tensors[self.name_embed_text]["input"][i].shape()))
        self.model.process(self.name_embed_text, self.tensors[self.name_embed_text]["input"], self.tensors[self.name_embed_text]["output"])

        if spk_idx != -1:
            for i in range(self.dev_ids_num):
                spk_emb_tensor = sail.Tensor(self.handle, [self.HIDDEN_SIZE], self.first_hidden_state_text[i].dtype(), True, False)
                spk_emb_tensor.update_data(np.array(spk_emb).astype(np.float16).view(np.uint16))
                self.first_hidden_state_text[i].sync_s2d(spk_emb_tensor, 0, spk_idx * self.HIDDEN_SIZE, self.HIDDEN_SIZE)
 
        # blocks
        for i in range(self.dev_ids_num):
            self.tensors[self.name_blocks[0]]["input"][3 * i + 1] = sail.Tensor(self.first_pid[i], [1, length], 0)
            self.tensors[self.name_blocks[0]]["input"][3 * i + 2] = sail.Tensor(self.first_attention_mask[i], [1, 1, length, length], 0)
            self.tensors[self.name_blocks[0]]["input"][3 * i + 1].update_data(position_id.reshape(self.tensors[self.name_blocks[0]]["input"][3 * i + 1].shape()))
            self.tensors[self.name_blocks[0]]["input"][3 * i + 2].update_data(attention_mask.reshape(self.tensors[self.name_blocks[0]]["input"][3 * i + 2].shape()).view(np.uint16))
        for i in range(self.NUM_LAYERS):
            for j in range(self.dev_ids_num):
                self.tensors[self.name_blocks[i]]["input"][3 * j] = sail.Tensor(self.first_hidden_state_text[j], [1, length, self.HIDDEN_SIZE], 0)
                self.tensors[self.name_blocks[i]]["output"][3 * j] = sail.Tensor(self.first_hidden_state_text[j], [1, length, self.HIDDEN_SIZE], 0)
                self.tensors[self.name_blocks[i]]["output"][3 * j + 1] = sail.Tensor(self.past_k[i][j], [1, length, self.ATTEN_HEAD, self.ATTEN_DIM], 0)
                self.tensors[self.name_blocks[i]]["output"][3 * j + 2] = sail.Tensor(self.past_v[i][j], [1, length, self.ATTEN_HEAD, self.ATTEN_DIM], 0)
            if i > 0:
                for j in range(self.dev_ids_num):
                    self.tensors[self.name_blocks[i]]["input"][3 * j + 1] = self.tensors[self.name_blocks[0]]["input"][3 * j + 1]
                    self.tensors[self.name_blocks[i]]["input"][3 * j + 2] = self.tensors[self.name_blocks[0]]["input"][3 * j + 2]
            # breakpoint()
            self.model.process(self.name_blocks[i], self.tensors[self.name_blocks[i]]["input"], self.tensors[self.name_blocks[i]]["output"])

        # breakpoint()
        # lm_head
        self.tensors[self.name_lm_code]["input"][0] = sail.Tensor(self.first_hidden_state_text[0], [1, 1, self.HIDDEN_SIZE], (self.code_token_length - 1) * self.HIDDEN_SIZE)
        self.tensors[self.name_lm_code]["output"][0] = self.lm_code_output[0]
        
        self.model.process(self.name_lm_code, self.tensors[self.name_lm_code]["input"], self.tensors[self.name_lm_code]["output"])

        last_hidden_tensor = sail.Tensor(self.handle, [self.HIDDEN_SIZE], self.first_hidden_state_text[0].dtype(), True, False)
        last_hidden_tensor.sync_d2s(self.first_hidden_state_text[0], (self.code_token_length - 1) * self.HIDDEN_SIZE, 0, self.HIDDEN_SIZE)
        last_hidden = last_hidden_tensor.asnumpy().view(np.float16)
        logits = self.lm_code_output[0].asnumpy()
        return logits, last_hidden
    
    def forward_next_code_core(self, tokens):
        self.code_token_length += 1
        position_id = np.array(self.code_token_length - 1, self.type_convert(self.tensors[self.name_blocks_cache[0]]["input"][1].dtype()))
        attention_mask = np.zeros(self.SEQLEN+1, self.type_convert(self.tensors[self.name_blocks_cache[0]]["input"][2].dtype()))
        for i in range(self.code_token_length - 1, self.SEQLEN):
            attention_mask[i] = self.ATTENTION_MASK

        # embedding_cache
        input_ids = np.array(tokens, dtype=self.type_convert(self.next_embed_input_code[0].dtype()))
        for i in range(self.dev_ids_num):
            self.next_embed_input_code[i].update_data(input_ids.reshape(self.tensors[self.name_embed_cache_code]["input"][i].shape()))
            self.tensors[self.name_embed_cache_code]["input"][i] = self.next_embed_input_code[i]
        for i in range(self.dev_ids_num):
            self.tensors[self.name_embed_cache_code]["output"][i] = self.next_hidden_state_code[i] 

        self.model.process(self.name_embed_cache_code, self.tensors[self.name_embed_cache_code]["input"], self.tensors[self.name_embed_cache_code]["output"])

        # block_cache
        for i in range(self.dev_ids_num):
            self.tensors[self.name_blocks_cache[0]]["input"][5 * i + 1] = self.next_pid[i]
            self.tensors[self.name_blocks_cache[0]]["input"][5 * i + 2] = self.next_attention_mask[i]
            self.tensors[self.name_blocks_cache[0]]["input"][5 * i + 1].update_data(position_id.reshape(self.tensors[self.name_blocks_cache[0]]["input"][5 * i + 1].shape()))
            self.tensors[self.name_blocks_cache[0]]["input"][5 * i + 2].update_data(attention_mask.reshape(self.tensors[self.name_blocks_cache[0]]["input"][5 * i + 2].shape()).view(np.uint16))

        for i in range(self.NUM_LAYERS):
            for j in range(self.dev_ids_num):
                # breakpoint()
                self.tensors[self.name_blocks_cache[i]]["input"][5 * j] = self.next_hidden_state_code[j]
                self.tensors[self.name_blocks_cache[i]]["output"][3 * j] = self.next_hidden_state_code[j]
                self.tensors[self.name_blocks_cache[i]]["input"][5 * j + 3] = self.past_k[i][j]
                self.tensors[self.name_blocks_cache[i]]["input"][5 * j + 4] = self.past_v[i][j]
                self.tensors[self.name_blocks_cache[i]]["output"][3 * j + 1] = self.past_kv_cache_tensors[i][j * 5 + 0][self.code_token_length]
                self.tensors[self.name_blocks_cache[i]]["output"][3 * j + 2] = self.past_kv_cache_tensors[i][j * 5 + 1][self.code_token_length]
            if i > 0:
                for j in range(self.dev_ids_num):
                    self.tensors[self.name_blocks_cache[i]]["input"][5 * j + 1] = self.tensors[self.name_blocks_cache[0]]["input"][5 * j + 1]
                    self.tensors[self.name_blocks_cache[i]]["input"][5 * j + 2] = self.tensors[self.name_blocks_cache[0]]["input"][5 * j + 2]
            # breakpoint()
            self.model.process(self.name_blocks_cache[i], self.tensors[self.name_blocks_cache[i]]["input"], self.tensors[self.name_blocks_cache[i]]["output"])

        
        #lm_head
        self.tensors[self.name_lm_code]["input"][0] = self.next_hidden_state_code[0]
        # breakpoint()
        self.tensors[self.name_lm_code]["output"][0] = self.lm_code_output[0]
        self.model.process(self.name_lm_code, self.tensors[self.name_lm_code]["input"], self.tensors[self.name_lm_code]["output"])

        last_hidden_tensor = sail.Tensor(self.handle, [self.HIDDEN_SIZE], self.next_hidden_state_code[0].dtype(), True, False)
        last_hidden_tensor.sync_d2s(self.next_hidden_state_code[0], 0, 0, self.HIDDEN_SIZE)
        last_hidden = last_hidden_tensor.asnumpy().view(np.float16)
        logits = self.lm_code_output[0].asnumpy()
        return logits, last_hidden