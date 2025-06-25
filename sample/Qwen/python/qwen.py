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
import os
import argparse
import readline

class Qwen:
    def __init__(self, config) -> None:
        self.version = "1.1.2"
        # read config file
        self.bmodel_path = config["bmodel_path"]
        tokenizer_path = config["token_path"]
        dev_ids = config.get("dev_ids", 0)
        self.enable_thinking = config.get("enable_thinking", True)
        self.generation_mode = config.get("generation_mode", "greedy")
        self.repeat_last_n = config.get("repeat_last_n", 32)
        self.temperature = config.get("temperature", 0.8)
        self.top_p = config.get("top_p", 0.8)
        self.repeat_penalty = config.get("repeat_penalty", 1.1)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        ID_IM_END = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        ID_END = self.tokenizer.convert_tokens_to_ids("<|end|>")
        EOF = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        self.EOS = [self.tokenizer.eos_token_id, ID_IM_END, ID_END, EOF]
        self.dev_ids = [int(x) for x in str(dev_ids).split(',')]
        self.handles = {dev: sail.Handle(dev) for dev in self.dev_ids}
        self.target = sail.Handle(self.dev_ids[0]).get_target()

        # load bmodel
        if self.target in ["BM1688", "CV186AH"]:
            self.model = sail.EngineLLM(self.bmodel_path, sail.BmrtFlag.BM_RUNTIME_SHARE_MEM, self.dev_ids)
            get_input_addr0 = self.model.create_max_input_tensors
            get_output_addr0 = self.model.create_max_output_tensors
        else:
            self.model = sail.EngineLLM(self.bmodel_path, self.dev_ids)
            get_input_addr0 = self.model.get_input_tensors_addrmode0
            get_output_addr0 = self.model.get_output_tensors_addrmode0

        self.tensors = {}
        self.graph_names = self.model.get_graph_names()
        self.io_alone = 0

        for net in self.graph_names:
            self.tensors[net] = {}
            self.tensors[net]["addr_mode"] = self.model.get_addr_mode(net)
            if self.tensors[net]["addr_mode"] == 0:
                self.tensors[net]['input'] = get_input_addr0(net)
                self.tensors[net]['output'] = get_output_addr0(net)
            elif self.tensors[net]["addr_mode"] == 1:
                self.io_alone = 1
                self.tensors[net]['input'] = self.model.get_input_tensors(net)
                self.tensors[net]['output'] = self.model.get_output_tensors(net)

        # initialize params
        self.is_dynamic = self.model.get_is_dynamic("block_0")
        self.embedding_dynamic = self.model.get_is_dynamic("embedding") if "embedding" in self.graph_names else False
        print("dynamic: ", self.is_dynamic)
        self.token_length = 0
        _, self.SEQLEN, self.HIDDEN_SIZE = self.tensors["block_0"]["input"][0].shape()
        _, _, self.ATTEN_HEAD, self.ATTEN_DIM = self.tensors["block_cache_0"]["input"][3].shape()
        self.NUM_LAYERS = sum(1 for item in self.graph_names if item.startswith("block_cache_"))
        self.tokens = []

        self.ATTENTION_MASK = -10000.0
        if self.tensors["block_0"]["input"][2].dtype() == sail.Dtype.BM_BFLOAT16:
            self.ATTENTION_MASK = 50716

        # initialize net name
        self.name_embed = "embedding"
        self.name_embed_cache = "embedding_cache"
        self.name_blocks = ["block_"+str(i) for i in range(self.NUM_LAYERS)]
        self.name_blocks_cache = ["block_cache_"+str(i) for i in range(self.NUM_LAYERS)]
        self.name_lm = "lm_head"
        self.greedy = "greedy_head"
        self.sample = "sample_head" if "sample_head" in self.graph_names else "penalty_sample_head"    # renamed by mlir

        if self.generation_mode == "greedy" and self.greedy in self.graph_names:
            print(f"Generation mode: {self.generation_mode}")
        elif self.generation_mode == "sample" and self.sample in self.graph_names:
            print(f"Generation mode: {self.generation_mode}")
        else:
            print(f"Generation mode: lmhead_with_topk")
            self.generation_mode = None

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
    

        if self.name_embed in self.tensors:
            self.first_embed_input = self.model.create_max_input_tensors(self.name_embed)
            self.first_hidden_state = self.model.create_max_output_tensors(self.name_embed)
            self.next_embed_input = self.model.create_max_input_tensors(self.name_embed_cache)
            self.next_hidden_state = self.model.create_max_output_tensors(self.name_embed_cache)
        else:
            self.first_hidden_state = {}
            self.next_hidden_state = {}
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

        if self.name_embed not in self.tensors:
            for i in range(len(self.dev_ids)):
                self.first_hidden_state[i] = self.init_tensor(self.dev_ids[i], self.tensors[self.name_blocks[0]]["input"][0])
                self.next_hidden_state[i] = self.init_tensor(self.dev_ids[i], self.tensors[self.name_blocks_cache[0]]["input"][0])
            self.embedding_path = os.path.dirname(self.bmodel_path) + "/embedding.bin"
            self.hidden_bytes = self.HIDDEN_SIZE*np.dtype(np.uint16).itemsize
            try:
                with open(self.embedding_path, "rb") as file:
                    self.embedding_content = file.read()
            except FileNotFoundError:
                raise RuntimeError("Unable to open embedding file")

    def load_and_infer_embedding(self, tokens):
        size = len(tokens)
        buffer = np.zeros((size, self.HIDDEN_SIZE), dtype=np.uint16)

        for i in range(min(size, self.token_length)):
            # 根据tokens的值定位到文件内容中的位置
            start_position = tokens[i] * self.hidden_bytes

            # 从读取的内存内容中提取数据
            data = self.embedding_content[start_position:start_position + self.hidden_bytes]

            if len(data) != self.hidden_bytes:
                raise RuntimeError("File read failed")

            buffer[i] = np.frombuffer(data, dtype=np.uint16)

        return buffer
    
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
        embedding_length = length if self.embedding_dynamic else self.SEQLEN
        if self.name_embed in self.tensors:
            input_ids = np.zeros(embedding_length, self.type_convert(self.tensors[self.name_embed]["input"][0].dtype()))
            input_ids[:len(token)] = token
        else:
            input_ids = np.zeros([embedding_length,self.HIDDEN_SIZE], self.type_convert(self.tensors[self.name_blocks[0]]["input"][0].dtype()))
            input_ids[:len(token)] = self.load_and_infer_embedding(token)

        position_id = np.zeros(length, self.type_convert(self.tensors[self.name_blocks[0]]["input"][1].dtype()))
        for i in range(self.token_length):
            position_id[i] = i

        attention_mask = np.ones(length*length, self.type_convert(self.tensors[self.name_blocks[0]]["input"][2].dtype())) * self.ATTENTION_MASK
        for i in range(len(token)):
            for j in range(length):
                if (j <= i):
                    attention_mask[i*length + j] = 0

        return input_ids, position_id, attention_mask
        
    def sample_token(self, length):
        # BM1688 or CV186AH or multi device: lmhead_with_topk( = lm_head + greedy_head )
        if self.generation_mode is None:
            return int(np.squeeze(self.tensors[self.name_lm]["output"][0].asnumpy()))
        # lm_head + greedy_head 贪心策略
        elif self.generation_mode == "greedy":
            self.tensors[self.greedy]["input"][0] = self.tensors[self.name_lm]["output"][0]
            self.model.process(self.greedy, self.tensors[self.greedy]["input"], self.tensors[self.greedy]["output"])
            return int(self.tensors[self.greedy]["output"][0].asnumpy())
        # lm_head + penalty_sample_head 重复惩罚策略
        elif self.generation_mode == "sample":
            self.tensors[self.sample]["input"][0] = self.tensors[self.name_lm]["output"][0]
            generated_tokens = np.ones([1, length], self.type_convert(self.tensors[self.sample]["input"][1].dtype())) * self.tokens[-1]
            repeat_last_n = min(self.repeat_last_n, self.token_length)
            generated_tokens[0, :repeat_last_n] = self.tokens[self.token_length - repeat_last_n : self.token_length]
            self.tensors[self.sample]["input"][1].update_data(generated_tokens)
            self.tensors[self.sample]["input"][2].update_data([self.top_p])
            self.tensors[self.sample]["input"][3].update_data([self.temperature])
            self.tensors[self.sample]["input"][4].update_data([self.repeat_penalty])
            self.model.process(self.sample, self.tensors[self.sample]["input"], self.tensors[self.sample]["output"])

            probs = self.tensors[self.sample]["output"][0].asnumpy()[0]
            token_TopK = self.tensors[self.sample]["output"][1].asnumpy()[0]
            return int(np.random.choice(token_TopK, p=probs / probs.sum()))
        else:
            raise ValueError("Invalid generation_mode parameter. Supported options are 'greedy' and 'sample'.")

    def forward_first(self):
        self.token_length = len(self.tokens)
        length = self.token_length + 1 if self.is_dynamic else self.SEQLEN
        embedding_length = length if self.embedding_dynamic else self.SEQLEN
        input_ids, position_id, attention_mask = self.get_first_input(length, self.tokens)

        # embedding
        if self.name_embed in self.tensors:
            for i in range(len(self.dev_ids)):
                # breakpoint()
                self.tensors[self.name_embed]["input"][i] = sail.Tensor(self.first_embed_input[i], [1, embedding_length], 0)
                self.tensors[self.name_embed]["output"][i] = sail.Tensor(self.first_hidden_state[i], [1, embedding_length, self.HIDDEN_SIZE], 0)
                self.tensors[self.name_embed]["input"][i].update_data(input_ids.reshape(self.tensors[self.name_embed]["input"][i].shape()))
            self.model.process(self.name_embed, self.tensors[self.name_embed]["input"], self.tensors[self.name_embed]["output"])
        else:
            for i in range(len(self.dev_ids)):
                self.first_hidden_state[i].update_data(input_ids.reshape(self.tensors[self.name_blocks[0]]["input"][0].shape()).view(np.uint16))

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
        
        # sample
        return self.sample_token(length)
    
    def forward_next(self):
        self.token_length += 1
        position_id = np.array(self.token_length - 1, self.type_convert(self.tensors[self.name_blocks_cache[0]]["input"][1].dtype()))
        attention_mask = np.zeros(self.SEQLEN+1, self.type_convert(self.tensors[self.name_blocks_cache[0]]["input"][2].dtype()))
        for i in range(self.token_length - 1, self.SEQLEN):
            attention_mask[i] = self.ATTENTION_MASK

        # embedding_cache
        if self.name_embed in self.tensors:            
            input_ids = np.array(self.tokens[-1], self.type_convert(self.tensors[self.name_embed_cache]["input"][0].dtype()))
            for i in range(len(self.dev_ids)):
                self.tensors[self.name_embed_cache]["input"][i] = sail.Tensor(self.next_embed_input[i], [1, 1], 0)
                self.tensors[self.name_embed_cache]["input"][i].update_data(input_ids.reshape(self.tensors[self.name_embed_cache]["input"][i].shape()))
                self.tensors[self.name_embed_cache]["output"][i] = self.next_hidden_state[i]

            self.model.process(self.name_embed_cache, self.tensors[self.name_embed_cache]["input"], self.tensors[self.name_embed_cache]["output"])
        else:
            temp_data = self.load_and_infer_embedding([int(self.tokens[-1])])
            temp_data = temp_data.reshape(self.tensors[self.name_blocks_cache[0]]["input"][0].shape()).view(np.uint16)
            for i in range(len(self.dev_ids)):
                self.next_hidden_state[i].update_data(temp_data)

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
        
        #lm_head
        self.tensors[self.name_lm]["input"][0] = self.next_hidden_state[0]
        # breakpoint()
        self.tensors[self.name_lm]["output"][0] = self.lm_output[0]
        self.model.process(self.name_lm, self.tensors[self.name_lm]["input"], self.tensors[self.name_lm]["output"])

        # sample
        return self.sample_token(self.SEQLEN)
    
    def chat_stream(self, messages):
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.enable_thinking)
        self.tokens = self.tokenizer(text).input_ids
        if (len(self.tokens) > self.SEQLEN - 5):
            yield f"##reach max length, max token length is {self.SEQLEN}"
            return
        first_start = time.time()
        token = self.forward_first()
        first_end = time.time()
        self.tokens.append(token)
        full_word_tokens = []
        tok_num = 0
        while token not in self.EOS and self.token_length < self.SEQLEN:
            full_word_tokens.append(token)
            word = self.tokenizer.decode(full_word_tokens)
            if "�" in word:
                token = self.forward_next()
                self.tokens.append(token)
                tok_num += 1
                continue
            yield word
            full_word_tokens = []
            token = self.forward_next()
            self.tokens.append(token)
            tok_num += 1
        next_end = time.time()
        print('\n\n')
        print(f"FTL: {(first_end - first_start):.3f} s")
        print(f"TPS: {(tok_num / (next_end - first_end)):.3f} token/s")

    def chat_stream_for_api(self, params):
        messages = [param.dict() for param in params]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.enable_thinking)
        self.tokens = self.tokenizer(text).input_ids
        if (len(self.tokens) > self.SEQLEN - 5):
            res_dict = {}
            res_dict["finish_reason"] = "length"
            res_dict["text"] = ""
            yield res_dict
            return
        token = self.forward_first()
        self.tokens.append(token)
        full_word_tokens = []
        while token not in self.EOS and self.token_length < self.SEQLEN:
            full_word_tokens.append(token)
            text = self.tokenizer.decode(full_word_tokens)
            if "�" in text:
                token = self.forward_next()
                self.tokens.append(token)
                continue
            res_dict = {}
            res_dict["finish_reason"] = None
            res_dict["text"] = text
            yield res_dict
            full_word_tokens = []
            token = self.forward_next()
            self.tokens.append(token)

    def chat_for_api(self, params):
        messages = [param.dict() for param in params]
        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.enable_thinking)
        self.tokens = self.tokenizer(input_text).input_ids
        if (len(self.tokens) > self.SEQLEN - 5):
            res_dict = {}
            res_dict["finish_reason"] = "length"
            res_dict["text"] = ""
            return res_dict
        all_token = []
        token = self.forward_first()
        self.tokens.append(token)
        while token not in self.EOS and self.token_length < self.SEQLEN:
            all_token.append(token)
            token = self.forward_next()
            self.tokens.append(token)
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
    qwen = Qwen(config)
    messages = []
    while True:
        input_str = input("\nQuestion: ")
        if input_str == "exit":
            break
        print("\nAnswer: ", end = '')
        if input_str == "clear":
            messages = []
            print('历史消息清除完毕')
        else:
            assistant_msg = ''
            messages.append({"role": "user", "content": input_str})
            for response in qwen.chat_stream(messages):
                assistant_msg += response
                print(response, flush=True, end='')
            messages.append({"role": "assistant", "content": assistant_msg})
            if ("##reach max length" in assistant_msg):
                messages = []
