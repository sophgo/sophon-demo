#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#

import argparse
import os
import sys
import time

import numpy as np
import sophon.sail as sail
import torch
import torchvision.transforms as T
from PIL import Image
from transformers import AutoTokenizer
import readline

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)
from preprocess import process_image, process_video

class InternVL3():
    def __init__(self, args):
        self.version = "1.0.0"
        # devid
        self.dev_ids = [args.devid]
        self.handles = {dev: sail.Handle(dev) for dev in self.dev_ids}
        self.system_prompt = '你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。'

        # load tokenizer
        print("Load " + args.config_path + " ...")
        self.tokenizer = AutoTokenizer.from_pretrained(args.config_path)

        # warm up
        self.tokenizer.decode([0])

        # load image
        self.EOS = [self.tokenizer.convert_tokens_to_ids('<|im_end|>')]

        # load model
        start_time = time.time()
        self.model = sail.EngineLLM(args.model_path, sail.BmrtFlag.BM_RUNTIME_SHARE_MEM, self.dev_ids)
        self.graph_names = self.model.get_graph_names()
        load_model_time = time.time() - start_time
        print(f"sail.EngineLLM init cost: {load_model_time:.3f} s")

        # initialize parameters
        self.target = sail.Handle(self.dev_ids[0]).get_target()
        self.tensors = {}
        start_time = time.time()
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
        init_tensor_time = time.time() - start_time
        print(f"io tensors init cost: {init_tensor_time:.3f} s")

        # initialize params
        self.is_dynamic = self.model.get_is_dynamic("block_0")
        print("dynamic: ", self.is_dynamic)
        self.token_length = 0
        _, self.SEQLEN, self.HIDDEN_SIZE = self.tensors["block_0"]["input"][0].shape()
        _, _, self.ATTEN_HEAD, self.ATTEN_DIM = self.tensors["block_cache_0"]["input"][3].shape()
        self.NUM_IMAGE_TOKEN, _ = self.tensors["vit"]["output"][0].shape()
        
        self.MEDIA_TOKEN_ID = 151667
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
        self.name_vit = "vit"

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
    
    def process_input(self, media_path):
        if media_path == "" or not os.path.exists(media_path):
            print(f"Can't find image or video: {media_path}, change to text mode")
            pixel_values = torch.tensor([])
            num_patches_list = []
            question = self.input_str
        else:
            VIDEO_EXTS = [".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".mpeg", ".mpg"]
            ext = os.path.splitext(media_path)[1].lower()
            if ext in VIDEO_EXTS:
                pixel_values, num_patches_list = process_video(media_path)
            else:
                pixel_values = process_image(media_path)
                num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []

            image_tags = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
            question = image_tags + self.input_str

        IMG_START_TOKEN = '<img>'
        IMG_END_TOKEN = '</img>'
        IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.NUM_IMAGE_TOKEN * num_patches + IMG_END_TOKEN
            question = question.replace('<image>', image_tokens, 1)

        prompt = (
            f'<|im_start|>system\n{self.system_prompt}<|im_end|>\n'
            f'<|im_start|>user\n{question}<|im_end|>\n'
            f'<|im_start|>assistant\n'
        )

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        return input_ids.flatten().numpy().astype(np.int32), pixel_values
    
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

        attention_mask = np.full((length, length), self.ATTENTION_MASK, dtype=self.type_convert(self.tensors[self.name_blocks[0]]["input"][2].dtype()))
        mask = np.tril(np.ones((length, length), dtype=bool))
        attention_mask[mask] = 0
        attention_mask = attention_mask.flatten()

        return input_ids, position_id, attention_mask
    
    def forward_first(self, tokens, pixel_values):
        self.token_length = len(tokens)
        if self.token_length > self.SEQLEN:
            print("warining, input seq len too large")
        length = self.token_length + 1 if self.is_dynamic else self.SEQLEN
        input_ids, position_id, attention_mask = self.get_first_input(length, tokens)
        visited_tokens = tokens.copy()

        for i in range(len(self.dev_ids)):
            self.tensors[self.name_embed]["input"][i] = sail.Tensor(self.first_embed_input[i], [1, length], 0)
            self.tensors[self.name_embed]["output"][i] = sail.Tensor(self.first_hidden_state[i], [1, length, self.HIDDEN_SIZE], 0)
            self.tensors[self.name_embed]["input"][i].update_data(input_ids.reshape(self.tensors[self.name_embed]["input"][i].shape()))
        self.model.process(self.name_embed, self.tensors[self.name_embed]["input"], self.tensors[self.name_embed]["output"])
        
        # ViT Inference
        self.vit_input = self.tensors[self.name_vit]["input"][0]
        self.vit_output = self.tensors[self.name_vit]["output"][0]
        if pixel_values is not None and pixel_values.numel() > 0:
            # 只有在提供了 pixel_values 时才进入原来的 Vit 推理
            vit_offset = -1
            for i in range(length):
                if visited_tokens[i] == self.MEDIA_TOKEN_ID:
                    vit_offset = i
                    break
            for i in range(pixel_values.shape[0]):
                if vit_offset >= 0 and pixel_values[i].numel() == np.prod(self.vit_input.shape()):
                    self.vit_input.update_data(pixel_values[i].unsqueeze(0).numpy())
                    self.vit_input.sync_s2d()
                    self.model.process(self.name_vit, {0: self.vit_input}, {0: self.vit_output})
                    # 把结果 copy 回 embedding output buffer
                    self.tensors[self.name_embed]["output"][0].sync_d2d(
                        self.vit_output, 0,
                        int((vit_offset + i * self.vit_output.shape()[0]) * self.HIDDEN_SIZE),
                        np.prod(self.vit_output.shape())
                    )

        # forward blocks
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
            self.model.process(self.name_blocks[i], self.tensors[self.name_blocks[i]]["input"], self.tensors[self.name_blocks[i]]["output"])

        # lm_head
        self.tensors[self.name_lm]["input"][0] = sail.Tensor(self.first_hidden_state[0], [1, 1, self.HIDDEN_SIZE], (self.token_length - 1) * self.HIDDEN_SIZE)
        self.tensors[self.name_lm]["output"][0] = self.lm_output[0]
        
        self.model.process(self.name_lm, self.tensors[self.name_lm]["input"], self.tensors[self.name_lm]["output"])
        if not self.is_sample:
            return (self.tensors[self.name_lm]["output"][0].asnumpy().item())

        # sample
        self.tensors[self.greedy]["input"][0] = self.tensors[self.name_lm]["output"][0]
        self.model.process(self.greedy, self.tensors[self.greedy]["input"], self.tensors[self.greedy]["output"])
        return (self.tensors[self.greedy]["output"][0].asnumpy().item())

    def forward_next(self):
        self.token_length += 1
        position_id = np.array(self.token_length - 1, self.type_convert(self.tensors[self.name_blocks_cache[0]]["input"][1].dtype()))
        attention_mask = np.zeros(self.SEQLEN+1, self.type_convert(self.tensors[self.name_blocks_cache[0]]["input"][2].dtype()))
        for i in range(self.token_length - 1, self.SEQLEN):
            attention_mask[i] = self.ATTENTION_MASK

        # embedding_cache
        if len(self.dev_ids) > 1:
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
            self.model.process(self.name_blocks_cache[i], self.tensors[self.name_blocks_cache[i]]["input"], self.tensors[self.name_blocks_cache[i]]["output"])
        
        #lm_head
        self.tensors[self.name_lm]["input"][0] = self.next_hidden_state[0]
        self.tensors[self.name_lm]["output"][0] = self.lm_output[0]
        self.model.process(self.name_lm, self.tensors[self.name_lm]["input"], self.tensors[self.name_lm]["output"])
        if not self.is_sample:
            return (self.tensors[self.name_lm]["output"][0].asnumpy().item())

        # sample
        self.tensors[self.greedy]["input"][0] = self.tensors[self.name_lm]["output"][0]
        self.model.process(self.greedy, self.tensors[self.greedy]["input"], self.tensors[self.greedy]["output"])

        return (self.tensors[self.greedy]["output"][0].asnumpy().item())
    
    def chat(self):
        """
        Start a chat session.
        """
        # Instruct
        print("""\n=================================================================
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

            media_path = input("\nImage or Video Path: ")
            media_path = media_path.strip()
            inputs = self.process_input(media_path)
            token_len = len(inputs[0])
            # check tokens
            if not self.input_str:
                print("Sorry: your question is empty!!")
                return
            if token_len > self.SEQLEN - 128:
                print(
                    "The maximum question length should be shorter than {} but we get {} instead.".format(
                        self.SEQLEN, token_len
                    )
                )
                continue

            print("\nAnswer: ", end="")
            self.stream_answer(inputs)



    def stream_answer(self, inputs):
        """
        Stream the answer for the given inputs.
        """
        tok_num = 0
        self.answer_token = []

        # First token
        first_start = time.time()
        token = self.forward_first(inputs[0], inputs[1])
        first_end = time.time()

        # Following tokens
        full_word_tokens = []
        while token not in self.EOS:
            self.answer_token.append(token)
            full_word_tokens.append(token)
            word = self.tokenizer.decode(full_word_tokens, skip_special_tokens=True)
            if "�" in word:
                token = self.forward_next()
                tok_num += 1
                continue
            print(word, flush=True, end="")

            token = self.forward_next()
            tok_num += 1
            full_word_tokens = []

        if tok_num >= self.SEQLEN:
            print("\n[Warning] Max token limit reached, stopping generation.")

        # counting time
        next_end = time.time()
        first_duration = first_end - first_start
        next_duration = next_end - first_end
        tps = tok_num / next_duration

        print()
        print(f"FTL: {first_duration:.3f} s")
        print(f"TPS: {tps:.3f} token/s")

def main(args):
    model = InternVL3(args)
    model.chat()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, default="../internvl3-2b_w4bf16_seq4096_bm1684x.bmodel", help='path to the bmodel file')
    parser.add_argument('-c', '--config_path', type=str, default="./config", help='path to the config file')
    parser.add_argument('-d', '--devid', type=int, default=0, help='device ID to use')
    args = parser.parse_args()
    main(args)


