#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#

import sophon.sail as sail
from transformers import AutoProcessor
import numpy as np
import yaml
import time
import argparse
from PIL import Image
import torch

class Llama:
    def __init__(self, bmodel_path, dev_ids, tokenizer_path, image_path) -> None:
        self.version = "1.0.0"
        self.processor = AutoProcessor.from_pretrained(tokenizer_path)
        self.tokenizer = self.processor.tokenizer
        self.EOS = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>"), 128001, 128008, 128009]
        self.dev_ids = [int(x) for x in str(dev_ids).split(',')]
        self.handles = {dev: sail.Handle(dev) for dev in self.dev_ids}
        self.image = Image.open(image_path)
        self.system = {"role": "user", "content": [{"type": "image"}]}
        # warm up
        self.tokenizer.decode([0])
        self.model = sail.EngineLLM(bmodel_path, self.dev_ids)
        self.tensors = {}
        self.graph_names = self.model.get_graph_names()
        self.io_alone = 0
        self.history = [self.system]
        self.enable_history = True

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
        print("dynamic: ", self.is_dynamic)
        _, self.SEQLEN, self.HIDDEN_SIZE = self.tensors["block_0"]["input"][0].shape()
        _, _, self.ATTEN_HEAD_PAST_K, self.ATTEN_DIM_PAST_K = self.tensors["block_cache_0"]["input"][3].shape()
        _, _, self.ATTEN_HEAD_PAST_V, self.ATTEN_DIM_PAST_V = self.tensors["block_cache_0"]["input"][4].shape()

        _, self.PARAMETER_PAST_K_CAL, self.ATTEN_HEAD_PAST_K_CAL, self.ATTEN_DIM_PAST_K_CAL = self.tensors["block_cache_3"]["input"][2].shape()
        _, self.PARAMETER_PAST_V_CAL, self.ATTEN_HEAD_PAST_V_CAL, self.ATTEN_DIM_PAST_V_CAL = self.tensors["block_cache_3"]["input"][3].shape()
        self.NUM_TILES, self.NUM_PATCHES, _ = self.tensors["vit"]["output"][0].shape()
        
        # self.ATTENTION_MASK = -9984
        if self.tensors["block_0"]["input"][2].dtype() == sail.Dtype.BM_BFLOAT16:
            self.ATTENTION_MASK = 49846

        self.is_sample = False
        if ("greedy_head" in self.graph_names):
            self.is_sample = True
        self.NUM_LAYERS = (len(self.graph_names) - 6) // 2
        self.token_length = 0
        self.cross_atten_layers = [3, 8, 13, 18, 23, 28, 33, 38]

        # initialize net name
        self.name_vit = "vit"
        self.name_embed = "embedding"
        self.name_embed_cache = "embedding_cache"
        self.name_blocks = ["block_"+str(i) for i in range(self.NUM_LAYERS)]
        self.name_blocks_cache = ["block_cache_"+str(i) for i in range(self.NUM_LAYERS)]
        self.name_lm = "lm_head"
        self.greedy = "greedy_head"
        self.penalty = "penalty_sample_head"

        self.past_k = {}
        self.past_v = {}
        # not io_alone 
        if self.io_alone == 0 or self.is_dynamic:
            print("no io_alone")
            for j in range(self.NUM_LAYERS):
                self.past_k[j] = self.init_tensor(self.dev_ids[0], self.tensors[self.name_blocks_cache[j]]["input"][3])
                self.past_v[j] = self.init_tensor(self.dev_ids[0], self.tensors[self.name_blocks_cache[j]]["input"][4])
        else:
            for j in range(self.NUM_LAYERS):
                if j in self.cross_atten_layers:
                    self.past_k[j] = self.tensors[self.name_blocks_cache[j]]["input"][2]
                    self.past_v[j] = self.tensors[self.name_blocks_cache[j]]["input"][3]
                else:
                    self.past_k[j] = self.tensors[self.name_blocks_cache[j]]["input"][3]
                    self.past_v[j] = self.tensors[self.name_blocks_cache[j]]["input"][4]

        self.pixel_values = self.init_tensor(self.dev_ids[0], self.tensors[self.name_vit]["input"][0])
        self.aspect_ratio_ids = self.init_tensor(self.dev_ids[0], self.tensors[self.name_vit]["input"][1])
        self.aspect_ratio_mask = self.init_tensor(self.dev_ids[0], self.tensors[self.name_vit]["input"][2])
        self.cross_attention_states_Reshape = self.init_tensor(self.dev_ids[0], self.tensors[self.name_vit]["output"][0])

        self.first_embed_input = self.model.create_max_input_tensors(self.name_embed)
        self.first_hidden_state = self.model.create_max_output_tensors(self.name_embed)
        self.next_embed_input = self.model.create_max_input_tensors(self.name_embed_cache)
        self.next_hidden_state = self.model.create_max_output_tensors(self.name_embed_cache)
        self.lm_input = self.model.create_max_input_tensors(self.name_lm)
        self.lm_output = self.model.create_max_output_tensors(self.name_lm)

        self.first_pid = self.init_tensor(self.dev_ids[0], self.tensors[self.name_blocks[0]]["input"][1])
        self.first_attention_mask = self.init_tensor(self.dev_ids[0], self.tensors[self.name_blocks[0]]["input"][2])
        
        self.next_pid = self.init_tensor(self.dev_ids[0], self.tensors[self.name_blocks_cache[0]]["input"][1])
        self.next_attention_mask = self.init_tensor(self.dev_ids[0], self.tensors[self.name_blocks_cache[0]]["input"][2])

    def init_input_tensor(self, dev_id, net, index):
        shape = self.model.get_input_shape(net, index)
        type = self.model.get_input_dtype(net, index)
        return sail.Tensor(self.handles[dev_id], shape, type, False, True) 
    
    def init_output_tensor(self, dev_id, net, index):
        shape = self.model.get_output_shape(net, index)
        type = self.model.get_output_dtype(net, index)
        return sail.Tensor(self.handles[dev_id], shape, type, False, True)
    
    def init_tensor_with_shape(self, dev_id, shape, type):
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
    
    def get_first_input(self, length, token):
        input_ids = np.zeros(length, self.type_convert(self.tensors[self.name_embed]["input"][0].dtype()))
        input_ids[:len(token)] = token

        position_id = np.zeros(length, self.type_convert(self.tensors[self.name_blocks[0]]["input"][1].dtype()))
        for i in range(self.token_length):
            position_id[i] = i

        attention_mask = np.ones(length*length, self.type_convert(self.tensors[self.name_blocks[0]]["input"][2].dtype())) * self.ATTENTION_MASK
        for i in range(len(token)):
            for j in range(length):
                if (j <= i):
                    attention_mask[i*length + j] = 0

        return input_ids, position_id, attention_mask
    

    def process_input(self):
        self.history.append({"role":"user","content":[{"type": "text", "text": self.input_str}]})
        input_text = self.processor.apply_chat_template(self.history, add_generation_prompt=True)
        inputs = self.processor(self.image, input_text, return_tensors="pt")
        for ins in inputs.keys():
            if inputs[ins].dtype == torch.int64:
                inputs[ins] = inputs[ins].to(torch.int32)
            inputs[ins] = inputs[ins].flatten().tolist()
        return inputs
        
    def forward_first(self, tokens, pixel_values, aspect_ratio_ids, aspect_ratio_mask, cross_atten_mask):
        tokens = tokens[-self.SEQLEN:] if len(tokens) > self.SEQLEN else tokens
        self.token_length = len(tokens)

        length = self.token_length + 1 if self.is_dynamic else self.SEQLEN
        input_ids, position_id, attention_mask = self.get_first_input(length, tokens)
        
        text_row_mask = np.zeros(self.SEQLEN).astype(np.uint16)
        text_row_mask[6:self.token_length] = 8064 #8064表示bf16的1

        cross_attention_mask = np.full((self.SEQLEN, self.NUM_TILES, self.NUM_PATCHES), self.ATTENTION_MASK).astype(np.uint16)
        cross_attention_mask[:6, :, :] = 0

        for i in range(6, self.token_length):
            for j in range(self.NUM_TILES):
                # cross_atten_mask 是一个一维数组，需要正确地索引
                if cross_atten_mask[i * self.NUM_TILES + j] == 1:
                    cross_attention_mask[i, j, :] = 0
                else:
                    cross_attention_mask[i, j, :] = self.ATTENTION_MASK

        # vit
        self.tensors[self.name_vit]["input"][0] = sail.Tensor(self.pixel_values, [4, 3, 560, 560], 0)
        self.tensors[self.name_vit]["input"][1] = sail.Tensor(self.aspect_ratio_ids, [1, 1], 0)
        self.tensors[self.name_vit]["input"][2] = sail.Tensor(self.aspect_ratio_mask, [1, 4], 0)
        self.tensors[self.name_vit]["output"][0] = sail.Tensor(self.cross_attention_states_Reshape, [self.NUM_TILES, self.NUM_PATCHES, self.HIDDEN_SIZE], 0)
        self.tensors[self.name_vit]["input"][0].update_data(np.array(pixel_values).reshape(self.tensors[self.name_vit]["input"][0].shape()))
        self.tensors[self.name_vit]["input"][1].update_data(np.array(aspect_ratio_ids).reshape(self.tensors[self.name_vit]["input"][1].shape()))
        self.tensors[self.name_vit]["input"][2].update_data(np.array(aspect_ratio_mask).reshape(self.tensors[self.name_vit]["input"][2].shape()))
        self.model.process(self.name_vit, self.tensors[self.name_vit]["input"], self.tensors[self.name_vit]["output"])

        # embedding
        self.tensors[self.name_embed]["input"][0] = sail.Tensor(self.first_embed_input[0], [1, length], 0)
        self.tensors[self.name_embed]["output"][0] = sail.Tensor(self.first_hidden_state[0], [1, length, self.HIDDEN_SIZE], 0)
        self.tensors[self.name_embed]["input"][0].update_data(input_ids.reshape(self.tensors[self.name_embed]["input"][0].shape()))
        self.model.process(self.name_embed, self.tensors[self.name_embed]["input"], self.tensors[self.name_embed]["output"])
 
        # blocks
        self.tensors[self.name_blocks[0]]["input"][1] = sail.Tensor(self.first_pid, [1, length], 0)
        self.tensors[self.name_blocks[0]]["input"][2] = sail.Tensor(self.first_attention_mask, [1, 1, length, length], 0)
        self.tensors[self.name_blocks[0]]["input"][1].update_data(position_id.reshape(self.tensors[self.name_blocks[0]]["input"][1].shape()))
        self.tensors[self.name_blocks[0]]["input"][2].update_data(attention_mask.reshape(self.tensors[self.name_blocks[0]]["input"][2].shape()).astype(np.uint16))
        
        self.tensors[self.name_blocks[self.cross_atten_layers[0]]]["input"][2] = self.init_tensor_with_shape(self.dev_ids[0], [1, length, 1], sail.Dtype.BM_BFLOAT16)
        self.tensors[self.name_blocks[self.cross_atten_layers[0]]]["input"][3] = self.init_tensor_with_shape(self.dev_ids[0], [1, 1, length, self.NUM_TILES * self.NUM_PATCHES], sail.Dtype.BM_BFLOAT16)
        self.tensors[self.name_blocks[self.cross_atten_layers[0]]]["input"][2].update_data(text_row_mask.reshape(self.tensors[self.name_blocks[self.cross_atten_layers[0]]]["input"][2].shape()))
        self.tensors[self.name_blocks[self.cross_atten_layers[0]]]["input"][3].update_data(cross_attention_mask.reshape(self.tensors[self.name_blocks[self.cross_atten_layers[0]]]["input"][3].shape()))

        for i in range(self.NUM_LAYERS):
            if i in self.cross_atten_layers:
                self.tensors[self.name_blocks[i]]["output"][0] = sail.Tensor(self.first_hidden_state[0], [1, length, self.HIDDEN_SIZE], 0)
                self.tensors[self.name_blocks[i]]["output"][1] = sail.Tensor(self.past_k[i], [1, self.PARAMETER_PAST_K_CAL, self.ATTEN_HEAD_PAST_K_CAL, self.ATTEN_DIM_PAST_K_CAL], 0)
                self.tensors[self.name_blocks[i]]["output"][2] = sail.Tensor(self.past_v[i], [1, self.PARAMETER_PAST_V_CAL, self.ATTEN_HEAD_PAST_V_CAL, self.ATTEN_DIM_PAST_V_CAL], 0)
                self.tensors[self.name_blocks[i]]["input"][0] = self.tensors[self.name_embed]["output"][0]
                self.tensors[self.name_blocks[i]]["input"][1] = self.tensors[self.name_vit]["output"][0]
                if i != self.cross_atten_layers[0]:
                    self.tensors[self.name_blocks[i]]["input"][2] = self.tensors[self.name_blocks[3]]["input"][2]
                    self.tensors[self.name_blocks[i]]["input"][3] = self.tensors[self.name_blocks[3]]["input"][3]
            else:
                self.tensors[self.name_blocks[i]]["output"][0] = sail.Tensor(self.first_hidden_state[0], [1, length, self.HIDDEN_SIZE], 0)
                self.tensors[self.name_blocks[i]]["output"][1] = sail.Tensor(self.past_k[i], [1, length, self.ATTEN_HEAD_PAST_K, self.ATTEN_DIM_PAST_K], 0)
                self.tensors[self.name_blocks[i]]["output"][2] = sail.Tensor(self.past_v[i], [1, length, self.ATTEN_HEAD_PAST_V, self.ATTEN_DIM_PAST_V], 0)
                self.tensors[self.name_blocks[i]]["input"][0] = self.tensors[self.name_embed]["output"][0]
                if i > 0:
                    self.tensors[self.name_blocks[i]]["input"][1] = self.tensors[self.name_blocks[0]]["input"][1]
                    self.tensors[self.name_blocks[i]]["input"][2] = self.tensors[self.name_blocks[0]]["input"][2]
            self.model.process(self.name_blocks[i], self.tensors[self.name_blocks[i]]["input"], self.tensors[self.name_blocks[i]]["output"])
        
        # lm_head
        self.tensors[self.name_lm]["input"][0] = sail.Tensor(self.first_hidden_state[0], [1, self.HIDDEN_SIZE], (self.token_length - 1) * self.HIDDEN_SIZE)
        self.tensors[self.name_lm]["output"][0] = self.lm_output[0]
        self.model.process(self.name_lm, self.tensors[self.name_lm]["input"], self.tensors[self.name_lm]["output"])
        if not self.is_sample:
            return int(self.tensors[self.name_lm]["output"][0].asnumpy())

        # sample
        self.tensors[self.greedy]["input"][0] = self.tensors[self.name_lm]["output"][0]
        self.model.process(self.greedy, self.tensors[self.greedy]["input"], self.tensors[self.greedy]["output"])

        return int(self.tensors[self.greedy]["output"][0].asnumpy())
    
    def forward_next(self):
        self.token_length += 1
        position_id = np.array(self.token_length - 1, self.type_convert(self.tensors[self.name_blocks_cache[0]]["input"][1].dtype()))

        cross_attention_mask = np.full((self.NUM_TILES * self.NUM_PATCHES,), self.ATTENTION_MASK).astype(np.uint16)
        tile_indices = np.repeat(np.arange(self.NUM_TILES), self.NUM_PATCHES).astype(np.uint16)
        cross_attention_mask[tile_indices < 2] = 0
        
        attention_mask = np.zeros(self.SEQLEN+1, self.type_convert(self.tensors[self.name_blocks_cache[0]]["input"][2].dtype()))
        for i in range(self.token_length - 1, self.SEQLEN):
            attention_mask[i] = self.ATTENTION_MASK

        # embedding_cache
        self.tensors[self.name_embed_cache]["input"][0] = self.tensors[self.name_lm]["output"][0]
        if self.is_sample:
            self.tensors[self.name_embed_cache]["input"][0] = self.tensors[self.greedy]["output"][0]
        self.tensors[self.name_embed_cache]["output"][0] = self.next_hidden_state[0]
        self.model.process(self.name_embed_cache, self.tensors[self.name_embed_cache]["input"], self.tensors[self.name_embed_cache]["output"])

        # block_cache
        self.tensors[self.name_blocks_cache[0]]["input"][1] = self.next_pid
        self.tensors[self.name_blocks_cache[0]]["input"][2] = self.next_attention_mask
        self.tensors[self.name_blocks_cache[0]]["input"][1].update_data(position_id.reshape(self.tensors[self.name_blocks_cache[0]]["input"][1].shape()))
        self.tensors[self.name_blocks_cache[0]]["input"][2].update_data(attention_mask.reshape(self.tensors[self.name_blocks_cache[0]]["input"][2].shape()).view(np.uint16))


        for i in range(self.NUM_LAYERS):
            if i in self.cross_atten_layers:
                self.tensors[self.name_blocks_cache[i]]["input"][0] = self.next_hidden_state[0]
                self.tensors[self.name_blocks_cache[i]]["input"][2] = self.past_k[i]
                self.tensors[self.name_blocks_cache[i]]["input"][3] = self.past_v[i]
                self.tensors[self.name_blocks_cache[i]]["output"][0] = self.next_hidden_state[0]
                if i == self.cross_atten_layers[0]:
                    self.tensors[self.name_blocks_cache[i]]["input"][1].update_data(cross_attention_mask.reshape(self.tensors[self.name_blocks_cache[i]]["input"][1].shape()))
                else:
                    self.tensors[self.name_blocks_cache[i]]["input"][1] = self.tensors[self.name_blocks_cache[self.cross_atten_layers[0]]]["input"][1]
            else:
                self.tensors[self.name_blocks_cache[i]]["input"][0] = self.next_hidden_state[0]
                self.tensors[self.name_blocks_cache[i]]["output"][0] = self.next_hidden_state[0]
                self.tensors[self.name_blocks_cache[i]]["input"][3] = self.past_k[i]
                self.tensors[self.name_blocks_cache[i]]["input"][4] = self.past_v[i]
                self.tensors[self.name_blocks_cache[i]]["output"][1] = sail.Tensor(self.past_k[i], [1, 1, self.ATTEN_HEAD_PAST_K, self.ATTEN_DIM_PAST_K], (self.token_length-1) * (self.ATTEN_HEAD_PAST_K * self.ATTEN_DIM_PAST_K))
                self.tensors[self.name_blocks_cache[i]]["output"][2] = sail.Tensor(self.past_v[i], [1, 1, self.ATTEN_HEAD_PAST_V, self.ATTEN_DIM_PAST_V], (self.token_length-1) * (self.ATTEN_HEAD_PAST_V * self.ATTEN_DIM_PAST_V))    
                if i > 0:
                    self.tensors[self.name_blocks_cache[i]]["input"][1] = self.tensors[self.name_blocks_cache[0]]["input"][1]
                    self.tensors[self.name_blocks_cache[i]]["input"][2] = self.tensors[self.name_blocks_cache[0]]["input"][2]
            self.model.process(self.name_blocks_cache[i], self.tensors[self.name_blocks_cache[i]]["input"], self.tensors[self.name_blocks_cache[i]]["output"])
             
        #lm_head
        self.tensors[self.name_lm]["input"][0] = self.next_hidden_state[0]
        self.tensors[self.name_lm]["output"][0] = self.lm_output[0]
        self.model.process(self.name_lm, self.tensors[self.name_lm]["input"], self.tensors[self.name_lm]["output"])
        if not self.is_sample:
            return int(self.tensors[self.name_lm]["output"][0].asnumpy())

        # sample
        self.tensors[self.greedy]["input"][0] = self.tensors[self.name_lm]["output"][0]
        self.model.process(self.greedy, self.tensors[self.greedy]["input"], self.tensors[self.greedy]["output"])

        return int(self.tensors[self.greedy]["output"][0].asnumpy())
    
    def clear(self):
        self.history = [self.system]

    def update_history(self):
        if self.token_length >= self.SEQLEN:
            print("... (reach the maximal length)", flush=True, end='')
            self.history = [self.system]
        else:
            self.history.append({"role":"assistant","content":self.answer_cur})

    def chat_stream(self, inputs):
        tok_num = 0
        self.answer_cur = ""
        self.answer_token = []
        first_start = time.time()
        # First token
        token = self.forward_first(inputs['input_ids'],
                                         inputs['pixel_values'],
                                         inputs['aspect_ratio_ids'],
                                         inputs['aspect_ratio_mask'],
                                         inputs['cross_attention_mask'])
        first_end = time.time()

        # Following tokens
        full_word_tokens = []
        while token not in self.EOS and self.token_length < self.SEQLEN:
            full_word_tokens.append(token)
            word = self.tokenizer.decode(full_word_tokens, skip_special_tokens=True)
            if "�" in word:
                token = self.forward_next()
                tok_num += 1
                continue

            self.answer_token += full_word_tokens
            print(word, flush=True, end="")

            token = self.forward_next()
            tok_num += 1
            full_word_tokens = []

        # counting time
        next_end = time.time()
        first_duration = first_end - first_start
        next_duration = next_end - first_end
        tps = tok_num / next_duration

        print()
        print(f"FTL: {first_duration:.3f} s")
        print(f"TPS: {tps:.3f} token/s")

        self.answer_cur = self.tokenizer.decode(self.answer_token)

        if self.enable_history:
            self.update_history()
        else:
            self.clear()

    
    def chat(self):
        print(
            """\n=================================================================
1. If you want to quit, please enter one of [q, quit, exit]
2. To create a new chat session or add new image, please enter one of [clear, new]
================================================================="""
)
        while True:
            self.input_str = input("\nQuestion: ")
            # Quit
            if self.input_str in ["exit", "q", "quit"]:
                break
            # New Chat
            elif self.input_str in ["clear", "new"]:
                image_path = input("\nNew image path:")
                try:
                    self.image = Image.open(image_path)
                    print(f'load new image:"{image_path}"')
                except:
                    print(f'load image:"{image_path}" faild, load origin image:"{args.image_path}" instead')
                self.clear()
            # Chat
            else:
                inputs = self.process_input()
                tokens = inputs['input_ids']
                # check tokens
                if not self.input_str:
                    print("Sorry: your question is empty!!")
                    return
                if len(tokens) > self.SEQLEN:
                    print(
                        "The maximum question length should be shorter than {} but we get {} instead.".format(
                            self.SEQLEN, len(tokens)
                        )
                    )
                    return

                print("\nAnswer: ", end="")
                self.chat_stream(inputs)


def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--config', type=str, default='./config/llama3.yaml', help='path of config file')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argsparser()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    llama = Llama(config["bmodel_path"], config["dev_ids"], config["token_path"], config["image_path"])
    llama.chat()
        