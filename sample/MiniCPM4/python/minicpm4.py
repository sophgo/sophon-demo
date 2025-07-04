#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#

import sophon.sail as sail
from transformers import AutoTokenizer,GenerationConfig
import numpy as np
import yaml
import time
import argparse

class MiniCPM4():
    def __init__(self, bmodel_path, dev_ids, tokenizer_path) -> None:

        print("Load " + tokenizer_path + " ...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

        # warm up
        self.tokenizer.decode([0])
        self.EOS = [self.tokenizer.eos_token_id]

        # load model
        self.dev_ids = [int(x) for x in str(dev_ids).split(',')]
        self.handles = {dev: sail.Handle(dev) for dev in self.dev_ids}
        self.target = sail.Handle(self.dev_ids[0]).get_target()
        if self.target in ["BM1688", "CV186AH"]:
            self.model = sail.EngineLLM(bmodel_path, sail.BmrtFlag.BM_RUNTIME_SHARE_MEM, self.dev_ids)
        else:
            self.model = sail.EngineLLM(bmodel_path, self.dev_ids)
        
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
                # breakpoint()
                if self.tensors[net]["addr_mode"] == 0:
                    self.tensors[net]['input'] = self.model.get_input_tensors_addrmode0(net)
                    self.tensors[net]['output'] = self.model.get_output_tensors_addrmode0(net)
                elif self.tensors[net]["addr_mode"] == 1:
                    self.io_alone = 1
                    self.tensors[net]['input'] = self.model.get_input_tensors(net)
                    self.tensors[net]['output'] = self.model.get_output_tensors(net)

        # initialize params
        self.is_dynamic = self.model.get_is_dynamic("block_0")
        # if dynamic, only prefill block is dynamic.
        print("dynamic: ", self.is_dynamic)
        self.token_length = 0
        _, self.SEQLEN, self.HIDDEN_SIZE = self.tensors["block_0"]["input"][0].shape()
        _, _, self.ATTEN_HEAD_PAST_K, self.ATTEN_DIM_PAST_K = self.tensors["block_cache_0"]["input"][3].shape()
        _, _, self.ATTEN_HEAD_PAST_V, self.ATTEN_DIM_PAST_V = self.tensors["block_cache_0"]["input"][4].shape()

        self.ATTENTION_MASK = 50716
        if self.tensors["block_0"]["input"][2].dtype() == sail.Dtype.BM_BFLOAT16:
            self.ATTENTION_MASK = 50716  # 0xC61C in uint16_t

        self.name_lm = "lm_head"
        self.lm_input = self.model.create_max_input_tensors(self.name_lm)
        self.lm_output = self.model.create_max_output_tensors(self.name_lm)


        self.stop_strings = []
        if self.lm_output[0].shape()[1] == 1:
            self.generation_mode = "lmhead_with_greedy"
            self.NUM_LAYERS = (len(self.graph_names)-3) // 2 
            print(f"lm_head output only one token, which has used greedy mode in bmodel")
        else:
            self.generation_mode = "greedy"
            self.NUM_LAYERS = (len(self.graph_names)-5) // 2 

        self.visited_tokens = []
        self.token_length = 0


        # initialize net name
        self.name_embed = "embedding"
        self.name_embed_cache = "embedding_cache"
        self.name_blocks = ["block_"+str(i) for i in range(self.NUM_LAYERS)]
        self.name_blocks_cache = ["block_cache_"+str(i) for i in range(self.NUM_LAYERS)]
        self.greedy = "greedy_head"
        self.penalty = "penalty_sample_head"

        self.past_k = {}
        self.past_v = {}
        for j in range(self.NUM_LAYERS):
            self.past_k[j] = {}
            self.past_v[j] = {}
            for i in range(len(self.dev_ids)):
                self.past_k[j][i] = self.tensors[self.name_blocks_cache[j]]["input"][5 * i + 3]
                self.past_k[j][i].zeros()
                self.past_v[j][i] = self.tensors[self.name_blocks_cache[j]]["input"][5 * i + 4]
                self.past_v[j][i].zeros()
    
        # embedding
        self.first_embed_input = self.model.create_max_input_tensors(self.name_embed)
        self.first_hidden_state = self.model.create_max_output_tensors(self.name_embed)

        # embedding_cache
        self.next_embed_input = self.model.create_max_input_tensors(self.name_embed_cache)
        self.next_hidden_state = self.model.create_max_output_tensors(self.name_embed_cache)

        self.first_pid = {}
        self.first_attention_mask = {}
        self.next_pid = {}
        self.next_attention_mask = {}
        for i in range(len(self.dev_ids)):
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
        input_ids = np.zeros(self.SEQLEN, self.type_convert(self.tensors[self.name_embed]["input"][0].dtype()))
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
        
    def forward_first(self, token):
        token = token[-self.SEQLEN:] if len(token) > self.SEQLEN else token
        self.token_length = len(token)
        length = self.token_length + 1 if self.is_dynamic else self.SEQLEN
        input_ids, position_id, attention_mask = self.get_first_input(length, token)

        # 存一个全局visited_tokens
        self.visited_tokens = token.copy()
        # breakpoint()

        for i in range(len(self.dev_ids)):
            self.tensors[self.name_embed]["input"][i] = self.first_embed_input[i]
            self.tensors[self.name_embed]["input"][i].update_data(input_ids.reshape(self.tensors[self.name_embed]["input"][i].shape()))

            self.tensors[self.name_embed]["output"][i] = sail.Tensor(self.first_hidden_state[i], [1, length, self.HIDDEN_SIZE], 0)
            
        self.model.process(self.name_embed, self.tensors[self.name_embed]["input"], self.tensors[self.name_embed]["output"])

 
        # blocks
        for i in range(len(self.dev_ids)):
            self.tensors[self.name_blocks[0]]["input"][3 * i + 1] = sail.Tensor(self.first_pid[i], [1, length], 0)
            self.tensors[self.name_blocks[0]]["input"][3 * i + 1].update_data(position_id.reshape(self.tensors[self.name_blocks[0]]["input"][3 * i + 1].shape()))

            self.tensors[self.name_blocks[0]]["input"][3 * i + 2] = sail.Tensor(self.first_attention_mask[i], [1, 1, length, length], 0)
            self.tensors[self.name_blocks[0]]["input"][3 * i + 2].update_data(attention_mask.reshape(self.tensors[self.name_blocks[0]]["input"][3 * i + 2].shape()).astype(np.uint16))
       
       
        for i in range(self.NUM_LAYERS):
            for j in range(len(self.dev_ids)):
                self.tensors[self.name_blocks[i]]["input"][3 * j] = sail.Tensor(self.first_hidden_state[j], [1, length, self.HIDDEN_SIZE], 0)
                self.tensors[self.name_blocks[i]]["output"][3 * j] = sail.Tensor(self.first_hidden_state[j], [1, length, self.HIDDEN_SIZE], 0)
                self.tensors[self.name_blocks[i]]["output"][3 * j + 1] = sail.Tensor(self.past_k[i][j], [1, length, self.ATTEN_HEAD_PAST_K, self.ATTEN_DIM_PAST_K], 0)
                self.tensors[self.name_blocks[i]]["output"][3 * j + 2] = sail.Tensor(self.past_v[i][j], [1, length, self.ATTEN_HEAD_PAST_V, self.ATTEN_DIM_PAST_V], 0)
            if i > 0:
                for j in range(len(self.dev_ids)):
                    self.tensors[self.name_blocks[i]]["input"][3 * j + 1] = self.tensors[self.name_blocks[0]]["input"][3 * j + 1]
                    self.tensors[self.name_blocks[i]]["input"][3 * j + 2] = self.tensors[self.name_blocks[0]]["input"][3 * j + 2]
            self.model.process(self.name_blocks[i], self.tensors[self.name_blocks[i]]["input"], self.tensors[self.name_blocks[i]]["output"])

        # lm_head
        self.tensors[self.name_lm]["input"][0] = sail.Tensor(self.first_hidden_state[0], [1, 1, self.HIDDEN_SIZE], (self.token_length - 1) * self.HIDDEN_SIZE)
        self.tensors[self.name_lm]["output"][0] = self.lm_output[0]
        
        self.model.process(self.name_lm, self.tensors[self.name_lm]["input"], self.tensors[self.name_lm]["output"])
        if self.generation_mode == "lmhead_with_greedy":
            return_token = int(self.tensors[self.name_lm]["output"][0].asnumpy()[0])
            
        # greedy
        elif self.generation_mode == "greedy":
            self.tensors[self.greedy]["input"][0] = self.tensors[self.name_lm]["output"][0]
            self.model.process(self.greedy, self.tensors[self.greedy]["input"], self.tensors[self.greedy]["output"])
            return_token = int(self.tensors[self.greedy]["output"][0].asnumpy()[0])
        
        else:
            raise ValueError(f"Unsupported generation mode: {self.generation_mode}.")
        
        
        self.visited_tokens.append(return_token)
        return return_token
    
    def forward_next(self):
        self.token_length += 1
        position_id = np.array(self.token_length - 1, self.type_convert(self.tensors[self.name_blocks_cache[0]]["input"][1].dtype()))
        attention_mask = np.zeros(self.SEQLEN+1, self.type_convert(self.tensors[self.name_blocks_cache[0]]["input"][2].dtype()))
        attention_mask[self.token_length - 1:self.SEQLEN] = self.ATTENTION_MASK

        # embedding_cache
        input_ids = np.array(self.visited_tokens[-1], self.type_convert(self.tensors[self.name_embed_cache]["input"][0].dtype()))
        for i in range(len(self.dev_ids)):
            self.next_embed_input[i].update_data(input_ids.reshape(self.tensors[self.name_embed_cache]["input"][i].shape()))
            self.tensors[self.name_embed_cache]["input"][i] = self.next_embed_input[i]
            self.tensors[self.name_embed_cache]["output"][i] = self.tensors[self.name_blocks_cache[0]]["input"][5 * i]
        self.model.process(self.name_embed_cache, self.tensors[self.name_embed_cache]["input"], self.tensors[self.name_embed_cache]["output"])


        # block_cache
        for i in range(len(self.dev_ids)):
            self.tensors[self.name_blocks_cache[0]]["input"][5 * i + 1] = self.next_pid[i]
            self.tensors[self.name_blocks_cache[0]]["input"][5 * i + 2] = self.next_attention_mask[i]
            self.tensors[self.name_blocks_cache[0]]["input"][5 * i + 1].update_data(position_id.reshape(self.tensors[self.name_blocks_cache[0]]["input"][5 * i + 1].shape()))
            self.tensors[self.name_blocks_cache[0]]["input"][5 * i + 2].update_data(attention_mask.reshape(self.tensors[self.name_blocks_cache[0]]["input"][5 * i + 2].shape()).view(np.uint16))

        for i in range(self.NUM_LAYERS):

            for j in range(len(self.dev_ids)):
                if i < self.NUM_LAYERS - 1:
                    self.tensors[self.name_blocks_cache[i]]["output"][3 * j] = self.tensors[self.name_blocks_cache[i+1]]["input"][5 * j]
                elif i == self.NUM_LAYERS - 1:
                    self.tensors[self.name_blocks_cache[i]]["output"][3 * j] = self.tensors[self.name_lm]["input"][0]

                self.tensors[self.name_blocks_cache[i]]["output"][3 * j + 1] = sail.Tensor(self.past_k[i][j], [1, 1, self.ATTEN_HEAD_PAST_K, self.ATTEN_DIM_PAST_K], (self.token_length-1) * (self.ATTEN_HEAD_PAST_K * self.ATTEN_DIM_PAST_K))
                self.tensors[self.name_blocks_cache[i]]["output"][3 * j + 2] = sail.Tensor(self.past_v[i][j], [1, 1, self.ATTEN_HEAD_PAST_V, self.ATTEN_DIM_PAST_V], (self.token_length-1) * (self.ATTEN_HEAD_PAST_V * self.ATTEN_DIM_PAST_V))

            if i > 0:
                for j in range(len(self.dev_ids)):
                    self.tensors[self.name_blocks_cache[i]]["input"][5 * j + 1] = self.tensors[self.name_blocks_cache[0]]["input"][5 * j + 1]
                    self.tensors[self.name_blocks_cache[i]]["input"][5 * j + 2] = self.tensors[self.name_blocks_cache[0]]["input"][5 * j + 2]

            self.model.process(self.name_blocks_cache[i], self.tensors[self.name_blocks_cache[i]]["input"], self.tensors[self.name_blocks_cache[i]]["output"])

        #lm_head
        self.tensors[self.name_lm]["output"][0] = self.lm_output[0]
        self.model.process(self.name_lm, self.tensors[self.name_lm]["input"], self.tensors[self.name_lm]["output"])
        if self.generation_mode == "lmhead_with_greedy":
            return_token = int(self.tensors[self.name_lm]["output"][0].asnumpy()[0])
        elif self.generation_mode == "greedy":
            self.tensors[self.greedy]["input"][0] = self.tensors[self.name_lm]["output"][0]
            self.model.process(self.greedy, self.tensors[self.greedy]["input"], self.tensors[self.greedy]["output"])
            return_token = int(self.tensors[self.greedy]["output"][0].asnumpy()[0])
        else:
            raise ValueError(f"Unsupported generation mode: {self.generation_mode}.")
        self.visited_tokens.append(return_token)

        return return_token



    def encode_tokens(self, messages):
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokens = self.tokenizer(text).input_ids
        return tokens

    def chat_stream(self, messages):
        tokens = self.encode_tokens(messages)
        if (len(tokens) > self.SEQLEN - 5):
            yield f"##reach max length, max token length is {self.SEQLEN}. history has been cleared. please try again."
        first_start = time.time()
        token = self.forward_first(tokens)
        first_end = time.time()
        answer_cur = ""
        tok_num = 0
        while(token not in self.EOS and self.token_length < self.SEQLEN):
            pre_word = self.tokenizer.decode([token], skip_special_tokens=True)
            word = self.tokenizer.decode([token, token], skip_special_tokens=True)[len(pre_word):]
            answer_cur += word
            if any(self.answer_cur.endswith(stop) for stop in self.stop_strings):
                break
            if "�" in word:
                token = self.forward_next()
                tok_num += 1
                continue
            yield word
            token = self.forward_next()
            tok_num += 1
        next_end = time.time()
        print(f"\nFTL: {(first_end - first_start):.3f} s")
        print(f"TPS: {(tok_num / (next_end - first_end)):.3f} token/s")
        

def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--config', type=str, default='./config/minicpm4.yaml', help='path of config file')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argsparser()
    system_prompt = "You are a helpful assistant."
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    minicpm = MiniCPM4(config["bmodel_path"], config["dev_ids"], config["token_path"])
    messages = [{"role": "system", "content": system_prompt}]
    print("""\n=================================================================
1. If you want to quit, please enter one of [q, quit, exit]
2. To create a new chat session, please enter one of [clear]
=================================================================\n""")
    while True:
        input_str = input("Question: ")
        if input_str.lower() in ["exit", "quit", "q"]:
            break
        elif input_str.lower() == "clear":
            messages = [{"role": "system", "content": system_prompt}]
            print("History has been cleared.")
            continue
        elif input_str.strip() == "":
            print("Input cannot be empty. Please try again.")
            continue
        else:
            print("\nAnswer: ", end = '')
            assistant_msg = ''
            messages.append({"role": "user", "content": input_str})
            for response in minicpm.chat_stream(messages):
                assistant_msg += response
                print(response, flush=True, end='')
            print("\n")
            messages.append({"role": "assistant", "content": assistant_msg})
            if ("##reach max length" in assistant_msg):
                messages = [{"role": "system", "content": system_prompt}]