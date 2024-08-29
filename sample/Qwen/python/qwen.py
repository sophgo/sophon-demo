#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#

import sophon.sail as sail
from transformers import AutoTokenizer
import numpy as np
import yaml
import time
import argparse

class Qwen:
    def __init__(self, bmodel_path, dev_ids, tokenizer_path) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.EOS = self.tokenizer.eos_token_id
        self.dev_ids = [int(x) for x in str(dev_ids).split(',')]
        self.handles = {dev: sail.Handle(dev) for dev in self.dev_ids}
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
        # self.is_dynamic = False
        print("dynamic: ", self.is_dynamic)
        self.token_length = 0
        _, self.SEQLEN, self.HIDDEN_SIZE = self.tensors["block_0"]["input"][0].shape()
        _, _, self.ATTEN_HEAD, self.ATTEN_DIM = self.tensors["block_cache_0"]["input"][3].shape()

        self.ATTENTION_MASK = -10000.0
        if self.tensors["block_0"]["input"][2].dtype() == sail.Dtype.BM_BFLOAT16:
            self.ATTENTION_MASK = 50716

        self.is_sample = False
        if ("greedy_head" in self.graph_names):
            self.is_sample = True
        self.NUM_LAYERS = (len(self.graph_names) - 3) // 2
        if self.is_sample:
            self.NUM_LAYERS = (len(self.graph_names) - 5) // 2
        self.token_length = 0


        # initialize net name
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
                self.past_k[j] = {}
                self.past_v[j] = {}
                for i in range(len(self.dev_ids)):
                    self.past_k[j][i] = self.init_tensor(self.dev_ids[i], self.tensors[self.name_blocks_cache[j]]["input"][5 * i + 3])
                    self.past_v[j][i] = self.init_tensor(self.dev_ids[i], self.tensors[self.name_blocks_cache[j]]["input"][5 * i + 4])
        else:
            for j in range(self.NUM_LAYERS):
                self.past_k[j] = {}
                self.past_v[j] = {}
                for i in range(len(self.dev_ids)):
                    self.past_k[j][i] = self.tensors[self.name_blocks_cache[j]]["input"][5 * i + 3]
                    self.past_v[j][i] = self.tensors[self.name_blocks_cache[j]]["input"][5 * i + 4]
    

        self.first_embed_input = self.model.create_max_input_tensors(self.name_embed)
        self.first_hidden_state = self.model.create_max_output_tensors(self.name_embed)
        self.next_embed_input = self.model.create_max_input_tensors(self.name_embed_cache)
        self.next_hidden_state = self.model.create_max_output_tensors(self.name_embed_cache)
        self.first_pid = {}
        self.next_pid = {}
        self.first_attention_mask = {}
        self.next_attention_mask = {}
        self.lm_input = self.model.create_max_input_tensors(self.name_lm)
        self.lm_output = self.model.create_max_output_tensors(self.name_lm)
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
        
    def forward_first(self, token):
        self.token_length = len(token)

        length = self.token_length + 1 if self.is_dynamic else self.SEQLEN
        # length = self.SEQLEN
        input_ids, position_id, attention_mask = self.get_first_input(length, token)

        for i in range(len(self.dev_ids)):
            # breakpoint()
            self.tensors[self.name_embed]["input"][i] = sail.Tensor(self.first_embed_input[i], [1, length], 0)
            self.tensors[self.name_embed]["output"][i] = sail.Tensor(self.first_hidden_state[i], [1, length, self.HIDDEN_SIZE], 0)
            self.tensors[self.name_embed]["input"][i].update_data(input_ids.reshape(self.tensors[self.name_embed]["input"][i].shape()))
        self.model.process(self.name_embed, self.tensors[self.name_embed]["input"], self.tensors[self.name_embed]["output"])

 
        # blocks
        for i in range(len(self.dev_ids)):
            self.tensors[self.name_blocks[0]]["input"][3 * i + 1] = sail.Tensor(self.first_pid[i], [1, length], 0)
            self.tensors[self.name_blocks[0]]["input"][3 * i + 2] = sail.Tensor(self.first_attention_mask[i], [1, 1, length, length], 0)
            self.tensors[self.name_blocks[0]]["input"][3 * i + 1].update_data(position_id.reshape(self.tensors[self.name_blocks[0]]["input"][3 * i + 1].shape()))
            self.tensors[self.name_blocks[0]]["input"][3 * i + 2].update_data(attention_mask.reshape(self.tensors[self.name_blocks[0]]["input"][3 * i + 2].shape()).view(np.uint16))
        for i in range(self.NUM_LAYERS):
            for j in range(len(self.dev_ids)):
                self.tensors[self.name_blocks[i]]["input"][3 * j] = sail.Tensor(self.first_hidden_state[j], [1, length, self.HIDDEN_SIZE], 0)
                self.tensors[self.name_blocks[i]]["output"][3 * j] = sail.Tensor(self.first_hidden_state[j], [1, length, self.HIDDEN_SIZE], 0)
                self.tensors[self.name_blocks[i]]["output"][3 * j + 1] = sail.Tensor(self.past_k[i][j], [1, length, self.ATTEN_HEAD, self.ATTEN_DIM], 0)
                self.tensors[self.name_blocks[i]]["output"][3 * j + 2] = sail.Tensor(self.past_v[i][j], [1, length, self.ATTEN_HEAD, self.ATTEN_DIM], 0)
            if i > 0:
                for j in range(len(self.dev_ids)):
                    self.tensors[self.name_blocks[i]]["input"][3 * j + 1] = self.tensors[self.name_blocks[0]]["input"][3 * j + 1]
                    self.tensors[self.name_blocks[i]]["input"][3 * j + 2] = self.tensors[self.name_blocks[0]]["input"][3 * j + 2]
            # breakpoint()
            self.model.process(self.name_blocks[i], self.tensors[self.name_blocks[i]]["input"], self.tensors[self.name_blocks[i]]["output"])

        # breakpoint()
        # lm_head
        self.tensors[self.name_lm]["input"][0] = sail.Tensor(self.first_hidden_state[0], [1, 1, self.HIDDEN_SIZE], (self.token_length - 1) * self.HIDDEN_SIZE)
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
        if len(self.dev_ids) > 1:
            # breakpoint()
            input_ids = np.array(int(self.tensors[self.name_lm]["output"][0].asnumpy()), self.type_convert(self.tensors[self.name_embed_cache]["input"][0].dtype()))
            for i in range(len(self.dev_ids)):
                self.next_embed_input[i].update_data(input_ids.reshape(self.tensors[self.name_embed_cache]["input"][i].shape()))
                self.tensors[self.name_embed_cache]["input"][i] = self.next_embed_input[i]
        else:
            self.tensors[self.name_embed_cache]["input"][0] = self.tensors[self.name_lm]["output"][0]
            if self.is_sample:
                self.tensors[self.name_embed_cache]["input"][0] = self.tensors[self.greedy]["output"][0]
        for i in range(len(self.dev_ids)):
            self.tensors[self.name_embed_cache]["output"][i] = self.next_hidden_state[i] 

        self.model.process(self.name_embed_cache, self.tensors[self.name_embed_cache]["input"], self.tensors[self.name_embed_cache]["output"])

        # block_cache
        for i in range(len(self.dev_ids)):
            self.tensors[self.name_blocks_cache[0]]["input"][5 * i + 1] = self.next_pid[i]
            self.tensors[self.name_blocks_cache[0]]["input"][5 * i + 2] = self.next_attention_mask[i]
            self.tensors[self.name_blocks_cache[0]]["input"][5 * i + 1].update_data(position_id.reshape(self.tensors[self.name_blocks_cache[0]]["input"][5 * i + 1].shape()))
            self.tensors[self.name_blocks_cache[0]]["input"][5 * i + 2].update_data(attention_mask.reshape(self.tensors[self.name_blocks_cache[0]]["input"][5 * i + 2].shape()).view(np.uint16))


        for i in range(self.NUM_LAYERS):
            for j in range(len(self.dev_ids)):
                # breakpoint()
                self.tensors[self.name_blocks_cache[i]]["input"][5 * j] = self.next_hidden_state[j]
                self.tensors[self.name_blocks_cache[i]]["output"][3 * j] = self.next_hidden_state[j]
                # self.tensors[self.name_blocks_cache[i]]["input"][5 * j + 3] = sail.Tensor(self.past_k[i][j], [1, self.HIDDEN_SIZE, shape[-2], shape[-1]], 0)
                # self.tensors[self.name_blocks_cache[i]]["input"][5 * j + 4] = sail.Tensor(self.past_v[i][j], [1, self.HIDDEN_SIZE, shape[-2], shape[-1]], 0)
                self.tensors[self.name_blocks_cache[i]]["input"][5 * j + 3] = self.past_k[i][j]
                self.tensors[self.name_blocks_cache[i]]["input"][5 * j + 4] = self.past_v[i][j]
                self.tensors[self.name_blocks_cache[i]]["output"][3 * j + 1] = sail.Tensor(self.past_k[i][j], [1, 1, self.ATTEN_HEAD, self.ATTEN_DIM], (self.token_length-1) * (self.ATTEN_HEAD * self.ATTEN_DIM))
                self.tensors[self.name_blocks_cache[i]]["output"][3 * j + 2] = sail.Tensor(self.past_v[i][j], [1, 1, self.ATTEN_HEAD, self.ATTEN_DIM], (self.token_length-1) * (self.ATTEN_HEAD * self.ATTEN_DIM))
            if i > 0:
                for j in range(len(self.dev_ids)):
                    self.tensors[self.name_blocks_cache[i]]["input"][5 * j + 1] = self.tensors[self.name_blocks_cache[0]]["input"][5 * j + 1]
                    self.tensors[self.name_blocks_cache[i]]["input"][5 * j + 2] = self.tensors[self.name_blocks_cache[0]]["input"][5 * j + 2]
            # breakpoint()
            self.model.process(self.name_blocks_cache[i], self.tensors[self.name_blocks_cache[i]]["input"], self.tensors[self.name_blocks_cache[i]]["output"])
            
            # shape = self.tensors[self.name_blocks_cache[i]]["output"][1].shape()
            # unit_size = shape[-1] * shape[-2]
            # for j in range(len(self.dev_ids)):
            #     self.tensors[self.name_blocks_cache[i]]["input"][5 * j + 3].sync_d2d(
            #         self.tensors[self.name_blocks_cache[i]]["output"][3 * j + 1],
            #         0,
            #         (self.token_length-1) * unit_size,
            #         unit_size
            #     )
            #     self.tensors[self.name_blocks_cache[i]]["input"][5 * j + 4].sync_d2d(
            #         self.tensors[self.name_blocks_cache[i]]["output"][3 * j + 2],
            #         0,
            #         (self.token_length-1) * unit_size,
            #         unit_size
            #     )

        
        #lm_head
        self.tensors[self.name_lm]["input"][0] = self.next_hidden_state[0]
        # breakpoint()
        self.tensors[self.name_lm]["output"][0] = self.lm_output[0]
        self.model.process(self.name_lm, self.tensors[self.name_lm]["input"], self.tensors[self.name_lm]["output"])
        if not self.is_sample:
            return int(self.tensors[self.name_lm]["output"][0].asnumpy())

        # sample
        self.tensors[self.greedy]["input"][0] = self.tensors[self.name_lm]["output"][0]
        self.model.process(self.greedy, self.tensors[self.greedy]["input"], self.tensors[self.greedy]["output"])

        return int(self.tensors[self.greedy]["output"][0].asnumpy())
    
    def chat_stream(self, messages):
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokens = self.tokenizer(text).input_ids
        if (len(tokens) > self.SEQLEN - 5):
            yield f"##reach max length, max token length is {self.SEQLEN}"
        first_start = time.time()
        token = self.forward_first(tokens)
        first_end = time.time()
        full_word_tokens = []
        tok_num = 0
        while(token != self.EOS and self.token_length < self.SEQLEN):
            full_word_tokens.append(token)
            word = self.tokenizer.decode(full_word_tokens)
            if "�" in word:
                token = self.forward_next()
                tok_num += 1
                continue
            yield word
            full_word_tokens = []
            token = self.forward_next()
            tok_num += 1
        next_end = time.time()
        print('\n\n')
        print(f"FTL: {(first_end - first_start):.3f} s")
        print(f"TPS: {(tok_num / (next_end - first_end)):.3f} token/s")

    def chat_stream_for_api(self, params):
        messages = [param.dict() for param in params]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokens = self.tokenizer(text).input_ids
        if (len(tokens) > self.SEQLEN - 5):
            res_dict = {}
            res_dict["finish_reason"] = "length"
            res_dict["text"] = ""
            yield res_dict
            return
        token = self.forward_first(tokens)
        full_word_tokens = []
        while(token != self.EOS and self.token_length < self.SEQLEN):
            full_word_tokens.append(token)
            text = self.tokenizer.decode(full_word_tokens)
            if "�" in text:
                token = self.forward_next()
                continue
            res_dict = {}
            res_dict["finish_reason"] = None
            res_dict["text"] = text
            yield res_dict
            full_word_tokens = []
            token = self.forward_next()

    def chat_for_api(self, params):
        messages = [param.dict() for param in params]
        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_tokens = self.tokenizer(input_text).input_ids
        if (len(input_tokens) > self.SEQLEN - 5):
            res_dict = {}
            res_dict["finish_reason"] = "length"
            res_dict["text"] = ""
            return res_dict
        all_token = []
        token = self.forward_first(input_tokens)
        while token != self.EOS and self.token_length < self.SEQLEN:
            all_token.append(token)
            token = self.forward_next()
        text = self.tokenizer.decode(all_token)
        res_dict = {}
        res_dict["finish_reason"] = "stop"
        res_dict["text"] = text
        return res_dict
def argsparser():

    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--config', type=str, default='./config/qwen.yaml', help='path of config file')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argsparser()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    qwen = Qwen(config["bmodel_path"], config["dev_ids"], config["token_path"])
    messages = []
    while True:
        input_str = input("\nQuestion: ")
        if input_str == "exit":
            break
        print("\nAnswer: ", end = '')
        assistant_msg = ''
        messages.append({"role": "user", "content": input_str})
        for response in qwen.chat_stream(messages):
            assistant_msg += response
            print(response, flush=True, end='')
        messages.append({"role": "assistant", "content": assistant_msg})
        if ("##reach max length" in assistant_msg):
            messages = []