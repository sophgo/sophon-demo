#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#

import sophon.sail as sail
import numpy as np
import yaml
import time
import argparse
from PIL import Image
import torch
from support.janus import VLChatProcessor
import readline


class Janus:
    def __init__(self, bmodel_path, dev_ids, processor_path, image_path) -> None:
        self.version = "1.0.0"
        self.processor = VLChatProcessor.from_pretrained(processor_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        # warm up
        self.tokenizer.decode([0])
        self.EOS = [self.tokenizer.eos_token_id]
        self.dev_ids = [int(x) for x in str(dev_ids).split(',')]
        self.handles = {dev: sail.Handle(dev) for dev in self.dev_ids}
        self.image_path = image_path
        self.image = Image.open(image_path).convert("RGB")
        
        self.model = sail.EngineLLM(bmodel_path, self.dev_ids)
        self.tensors = {}
        self.graph_names = self.model.get_graph_names()
        self.io_alone = 0

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

        self.NUM_TILES, self.NUM_PATCHES, _ = self.tensors["vit"]["output"][0].shape()
        
        self.ATTENTION_MASK = -10000.0
        if self.tensors["block_0"]["input"][2].dtype() == sail.Dtype.BM_BFLOAT16:
            self.ATTENTION_MASK = 50716

        self.is_sample = False
        if ("greedy_head" in self.graph_names):
            self.is_sample = True
        self.NUM_LAYERS = (len(self.graph_names) - 6) // 2
        self.token_length = 0

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
                    self.past_k[j] = self.tensors[self.name_blocks_cache[j]]["input"][3]
                    self.past_v[j] = self.tensors[self.name_blocks_cache[j]]["input"][4]

        self.pixel_values = self.init_tensor(self.dev_ids[0], self.tensors[self.name_vit]["input"][0])
        self.vit_output = self.init_tensor(self.dev_ids[0], self.tensors[self.name_vit]["output"][0])

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
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{self.input_str}",
                "images": [self.image_path],
            },
            {
                "role": "<|Assistant|>",
                "content": ""
            },
        ]
        inputs = self.processor(
            conversations=conversation,
            images=[self.image]
        )
        inputs['input_ids'] = inputs['input_ids'].to(torch.int32).flatten().tolist()
        inputs['pixel_values'] = inputs['pixel_values'].flatten().tolist()
        return inputs
        
    def forward_first(self, tokens, pixel_values):
        tokens = tokens[-self.SEQLEN:] if len(tokens) > self.SEQLEN else tokens
        self.token_length = len(tokens)

        length = self.token_length + 1 if self.is_dynamic else self.SEQLEN
        input_ids, position_id, attention_mask = self.get_first_input(length, tokens)

        # embedding
        self.tensors[self.name_embed]["input"][0] = sail.Tensor(self.first_embed_input[0], [1, length], 0)
        self.tensors[self.name_embed]["output"][0] = sail.Tensor(self.first_hidden_state[0], [1, length, self.HIDDEN_SIZE], 0)
        self.tensors[self.name_embed]["input"][0].update_data(input_ids.reshape(self.tensors[self.name_embed]["input"][0].shape()))
        self.model.process(self.name_embed, self.tensors[self.name_embed]["input"], self.tensors[self.name_embed]["output"])

        # vit
        self.tensors[self.name_vit]["input"][0] = sail.Tensor(self.pixel_values, [1, 3, 384, 384], 0)
        self.tensors[self.name_vit]["output"][0] = sail.Tensor(self.vit_output, [self.NUM_TILES, self.NUM_PATCHES, self.HIDDEN_SIZE], 0)
        self.tensors[self.name_vit]["input"][0].update_data(np.array(pixel_values).reshape(self.tensors[self.name_vit]["input"][0].shape()))
        self.model.process(self.name_vit, self.tensors[self.name_vit]["input"], self.tensors[self.name_vit]["output"])
        
        # blocks
        unit_size = self.vit_output.shape()[-1]
        self.tensors[self.name_embed]["output"][0].sync_d2d(self.tensors[self.name_vit]["output"][0], 0, 42 * unit_size, 576 * unit_size)
        self.tensors[self.name_blocks[0]]["input"][1] = sail.Tensor(self.first_pid, [1, length], 0)
        self.tensors[self.name_blocks[0]]["input"][2] = sail.Tensor(self.first_attention_mask, [1, 1, length, length], 0)
        self.tensors[self.name_blocks[0]]["input"][1].update_data(position_id.reshape(self.tensors[self.name_blocks[0]]["input"][1].shape()))
        self.tensors[self.name_blocks[0]]["input"][2].update_data(attention_mask.reshape(self.tensors[self.name_blocks[0]]["input"][2].shape()).astype(np.uint16))
        
        for i in range(self.NUM_LAYERS):
            self.tensors[self.name_blocks[i]]["input"][0] = sail.Tensor(self.first_hidden_state[0],[1, length, self.HIDDEN_SIZE], 0)
            self.tensors[self.name_blocks[i]]["output"][0] = sail.Tensor(self.first_hidden_state[0], [1, length, self.HIDDEN_SIZE], 0)
            self.tensors[self.name_blocks[i]]["output"][1] = sail.Tensor(self.past_k[i], [1, length, self.ATTEN_HEAD_PAST_K, self.ATTEN_DIM_PAST_K], 0)
            self.tensors[self.name_blocks[i]]["output"][2] = sail.Tensor(self.past_v[i], [1, length, self.ATTEN_HEAD_PAST_V, self.ATTEN_DIM_PAST_V], 0)
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

    def chat_stream(self, inputs):
        tok_num = 0
        self.answer_cur = ""
        self.answer_token = []
        first_start = time.time()
        # First token
        token = self.forward_first(inputs['input_ids'], inputs['pixel_values'])
        first_end = time.time()

        # Following tokens
        while token not in self.EOS and self.token_length < self.SEQLEN:
            word = self.tokenizer.decode(token, skip_special_tokens=True)
            print(word, flush=True, end="")
            token = self.forward_next()
            tok_num += 1

        # counting time
        next_end = time.time()
        first_duration = first_end - first_start
        next_duration = next_end - first_end
        tps = tok_num / next_duration

        print()
        print(f"FTL: {first_duration:.3f} s")
        print(f"TPS: {tps:.3f} token/s")
    
    def chat(self):
        """
        Start a chat session.
        """
        # Instruct
        print(
"""\n=================================================================
1. If you want to quit, please enter one of [q, quit, exit]
2. To create a new chat session, please enter one of [clear, new]
================================================================="""
)
        # Stop Chatting with "exit" input
        while True:
            self.input_str = input("\nQuestion: ")
            # Quit
            if self.input_str in ["exit", "q", "quit"]:
                break
            # New Chat
            elif self.input_str in ["clear", "new"]:
                image_path = input("\nNew image path:")
                try:
                    self.image = Image.open(image_path).convert("RGB")
                    print(f'load new image:"{image_path}"')
                except:
                    print(f'load image:"{image_path}" faild, load origin image:"{self.image_path}" instead')
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
    parser.add_argument('--config', type=str, default='./config/janus.yaml', help='path of config file')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argsparser()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    janus = Janus(config["bmodel_path"], config["dev_ids"], config["token_path"], config["image_path"])
    janus.chat()
        