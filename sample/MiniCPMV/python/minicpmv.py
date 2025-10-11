# ===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ===----------------------------------------------------------------------===#

import sophon.sail as sail
from transformers import AutoProcessor, AutoConfig
import numpy as np
import yaml
import time
import argparse
import os
import torch
import torch.nn.functional as F
from PIL import Image
from copy import deepcopy
import json
from decord import VideoReader, cpu

MAX_NUM_FRAMES = 64

def encode_video(video_path):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    print('num frames:', len(frames))
    return frames

class MiniCPMV():
    def __init__(self, bmodel_path, dev_ids, tokenizer_path, input_path) -> None:
        self.version = "1.0.0"
        self.input_path = input_path
        self.dev_ids = [int(x) for x in str(dev_ids).split(',')]
        self.handles = {dev: sail.Handle(dev) for dev in self.dev_ids}
        self.bm_handle = self.handles[self.dev_ids[0]]
        self.target = self.bm_handle.get_target()

        print("Load " + bmodel_path + " ...")
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
        # HIDDEN_SIZE 2560 MAX_PATCHES 4900  VIT_DIMS 588 VIT_HIDDEN_SIZE 1 MAX_PIXELS 960400
        _, self.SEQLEN, self.HIDDEN_SIZE = self.tensors["block_0"]["input"][0].shape()
        _, _, self.ATTEN_HEAD_PAST_K, self.ATTEN_DIM_PAST_K = self.tensors["block_cache_0"]["input"][3].shape()
        _, _, self.ATTEN_HEAD_PAST_V, self.ATTEN_DIM_PAST_V = self.tensors["block_cache_0"]["input"][4].shape()
        self.MAX_PATCHES, self.VIT_DIMS = self.tensors["vit"]["input"][0].shape()
        # _, self.VIT_HIDDEN_SIZE = self.tensors["vit"]["input"][3].shape()
        self.VIT_HIDDEN_SIZE = 1
        self.MAX_PIXELS = self.MAX_PATCHES * 14 * 14

        print("Load " + tokenizer_path + " ...")
        self.processor = AutoProcessor.from_pretrained(tokenizer_path,
                                                       trust_remote_code=True,
                                                       size=None,
                                                       max_pixels=self.MAX_PIXELS,
                                                       min_pixels=64 * 28 * 28)
        self.tokenizer = self.processor.tokenizer
        # warm up
        self.tokenizer.decode([0])

        self.ID_END = self.tokenizer.convert_tokens_to_ids('</s>')
        self.ID_START = self.tokenizer.convert_tokens_to_ids('<s>')
        self.ID_IMAGE_START = self.tokenizer.convert_tokens_to_ids('<|im_start|>')
        self.ID_IMAGE_END = self.tokenizer.convert_tokens_to_ids('<|im_end|>')
        
        self.vision_config = AutoConfig.from_pretrained(tokenizer_path, trust_remote_code=True).vision_config
        self.hidden_size = self.vision_config.hidden_size
        self.num_patches_per_side = self.vision_config.image_size // self.vision_config.patch_size

        if self.tensors["lm_head"]["output"][0].shape()[1] == 1:
            self.generation_mode = "lmhead_with_topk"
        else:
            self.generation_mode = "greedy"

        self.ATTENTION_MASK = 50716
        if self.tensors["block_0"]["input"][2].dtype() == sail.Dtype.BM_BFLOAT16:
            self.ATTENTION_MASK = 50716  # 0xC61C in uint16_t
        
                # initialize net name
        self.name_embed = "embedding"
        self.name_embed_cache = "embedding_cache"

        self.stop_strings = []
        self.NUM_LAYERS = (len(self.graph_names)-4) // 2 

        self.visited_tokens = []
        self.token_length = 0

        self.name_blocks = ["block_"+str(i) for i in range(self.NUM_LAYERS)]
        self.name_blocks_cache = ["block_cache_"+str(i) for i in range(self.NUM_LAYERS)]
        self.name_vit = "vit"
        self.greedy = "greedy_head"
        self.penalty = "penalty_sample_head"

        # embedding
        self.first_embed_input = self.model.create_max_input_tensors(self.name_embed)
        self.first_hidden_state = self.model.create_max_output_tensors(self.name_embed)

        # embedding_cache
        self.next_embed_input = self.model.create_max_input_tensors(self.name_embed_cache)
        self.next_hidden_state = self.model.create_max_output_tensors(self.name_embed_cache)

        # vit
        self.vit_input_states = self.init_tensor(self.dev_ids[0], self.tensors[self.name_vit]["input"][0])
        self.vit_pos_ids = self.init_tensor(self.dev_ids[0], self.tensors[self.name_vit]["input"][1])
        self.vit_attn_mask = self.init_tensor(self.dev_ids[0], self.tensors[self.name_vit]["input"][2])
        self.vit_pos_embed_ids = self.init_tensor(self.dev_ids[0], self.tensors[self.name_vit]["input"][3])
        self.vit_resampler_attn_mask = self.init_tensor(self.dev_ids[0], self.tensors[self.name_vit]["input"][4])
        self.vit_out_tensor = self.init_tensor(self.dev_ids[0], self.tensors[self.name_vit]["output"][0])

        # lm_head
        self.name_lm = "lm_head"
        self.lm_input = self.model.create_max_input_tensors(self.name_lm)
        self.lm_output = self.model.create_max_output_tensors(self.name_lm)

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

    def process_message(self, image_list, msgs, media_type):
        prompts_lists = []
        input_images_lists = []
        if isinstance(msgs, str):
            msgs = json.loads(msgs)
        copy_msgs = deepcopy(msgs)

        if image_list is not None and isinstance(copy_msgs[0]["content"], str):
            copy_msgs[0]["content"] = [image_list, copy_msgs[0]["content"]]

        images = []
        for i, msg in enumerate(copy_msgs):
            role = msg["role"]
            content = msg["content"]
            assert role in ["user", "assistant"]
            if i == 0:
                assert role == "user", "The role of first msg should be user"
            if isinstance(content, str):
                content = [content]
            cur_msgs = []
            for c in content:
                if isinstance(c, Image.Image):
                    images.append(c)
                    cur_msgs.append("(<image>./</image>)")
                elif isinstance(c, str):
                    cur_msgs.append(c)
            msg["content"] = "\n".join(cur_msgs)

        prompt = self.processor.tokenizer.apply_chat_template(copy_msgs, tokenize=False, add_generation_prompt=True)
        prompts_lists.append(prompt)
        input_images_lists.append(images)

        max_slice_nums = None
        use_image_id = None
        if media_type == "video":
            max_slice_nums = 2
            use_image_id = False
        max_inp_length = 32768
        inputs = self.processor(prompts_lists,
                                input_images_lists, 
                                max_slice_nums=max_slice_nums,
                                use_image_id=use_image_id,
                                return_tensors="pt",
                                max_length=max_inp_length)
        return inputs
    
    def text_message(self):
        messages = [{"role": "user", "content": [self.input_str]}]
        return messages

    def image_message(self, path):
        image = Image.open(path).convert('RGB')
        messages = [{"role": "user", "content": [image, self.input_str]}]
        return image, messages

    def video_message(self, path):
        frames = encode_video(path)
        messages = [{"role": "user", "content": frames + [self.input_str]}]
        return frames, messages
    
    def get_media_type(self, file_path):
        image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        if ext in image_exts:
            return "image"
        if ext in video_exts:
            return "video"
        raise RuntimeError(f"Unsupported media type: {ext}")
        
    def vit_process_image(self, inputs):
        pixel_values = inputs.pixel_values[0] # batch = 1
        image_bound = inputs.image_bound[0]
        tgt_sizes = inputs.tgt_sizes[0]
        for i, pixel_value in  enumerate(pixel_values):
            tgt_size = tgt_sizes[i].numpy()
            pixel_value = pixel_value.numpy()
            pixel_value = np.transpose(pixel_value.reshape(3, 14, -1, 14), [2, 0, 1, 3])
            hidden_states = np.ascontiguousarray(pixel_value.reshape(-1, 3 * 14 * 14))
            cu_seqlens = torch.tensor([tgt_size[0] * tgt_size[1]], dtype=torch.int32)
            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
            seq_len, _ = hidden_states.shape
            full_mask = self.get_attn_mask(seq_len, cu_seqlens)
            vit_offset = image_bound[i][0].item()
            position_ids = self.get_position_ids(tgt_size)
            pos_embed_ids = self.get_pos_embed_ids(tgt_size)
            self.forward_vit(hidden_states, position_ids, full_mask, pos_embed_ids,
                                    tgt_size, vit_offset)

    def vit_process_video(self, inputs):
        self.vit_process_image(inputs)

    def get_attn_mask(self, seq_length, cu_seqlens):
        attention_mask = torch.full([1, seq_length, seq_length], -10000.0, dtype=torch.float32)
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1]:cu_seqlens[i],
                           cu_seqlens[i - 1]:cu_seqlens[i]] = 0
        return attention_mask.numpy().astype(np.float32)

    def get_position_ids(self, tgt_size: np.array):
        tgt_size = tgt_size.tolist()
        position_ids = torch.full(
            size=(
                tgt_size[0] * tgt_size[1],
            ),
            fill_value=0,
        )
        nb_patches_h = tgt_size[0]
        nb_patches_w = tgt_size[1]
        boundaries = torch.arange(1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side)
        fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_patches_h)
        fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_patches_w)
        bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
        bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)
        pos_ids = (bucket_coords_h[:, None] * self.num_patches_per_side + bucket_coords_w).flatten()
        position_ids = pos_ids
        return position_ids.numpy().astype(np.int32)
    
    def get_pos_embed_ids(self, tgt_size):
        pos_embed_ids = []
        height, width = tgt_size
        for h in range(height):
            for w in range(width):
                pos_id = h * self.num_patches_per_side + w
                pos_embed_ids.append(pos_id)
        pos_embed_ids = np.array(pos_embed_ids, dtype=np.int32)
        return pos_embed_ids


    def forward_embed(self, token):
        token = token[-self.SEQLEN:] if len(token) > self.SEQLEN else token
        self.token_length = len(token)
        length = self.token_length + 1 if self.is_dynamic else self.SEQLEN

        input_ids = np.zeros(self.SEQLEN, dtype=self.type_convert(self.tensors[self.name_embed]["input"][0].dtype()))
        input_ids[:self.token_length] = token

        # 存一个全局visited_tokens
        self.visited_tokens = token.copy()

        for i in range(len(self.dev_ids)):
            self.tensors[self.name_embed]["input"][i] = self.first_embed_input[i]
            self.tensors[self.name_embed]["input"][i].update_data(input_ids.reshape(self.tensors[self.name_embed]["input"][i].shape()))
            self.tensors[self.name_embed]["output"][i] = sail.Tensor(self.first_hidden_state[i], [1, length, self.HIDDEN_SIZE], 0)
        self.model.process(self.name_embed, self.tensors[self.name_embed]["input"], self.tensors[self.name_embed]["output"])
        
    def forward_vit(self, pixel_values, position_ids, full_attn_mask, pos_embed_ids, grid_hw, vit_offset):
        h, w = grid_hw[0], grid_hw[1]
        hw = h * w
        vit_input_states_shape = (4900, 588)
        position_ids_shape = 4900
        pos_embed_ids_shape = 4900
        p_pixel_values = np.pad(pixel_values, ((0, vit_input_states_shape[0] - pixel_values.shape[0]), (0, 0)), mode='constant')
        p_position_ids = np.pad(position_ids, ((0, position_ids_shape - position_ids.shape[0])), mode='constant')
        p_pos_embed_ids = np.pad(pos_embed_ids, ((0, pos_embed_ids_shape - pos_embed_ids.shape[0])), mode='constant')


        self.tensors[self.name_vit]["input"][0] = sail.Tensor(self.vit_input_states, [4900, 588], 0)
        self.tensors[self.name_vit]["input"][1] = sail.Tensor(self.vit_pos_ids, [4900], 0)
        self.tensors[self.name_vit]["input"][2] = sail.Tensor(self.vit_attn_mask, [1, 1, 4900, 4900], 0)
        self.tensors[self.name_vit]["input"][3] = sail.Tensor(self.vit_pos_embed_ids, [4900], 0)
        self.tensors[self.name_vit]["input"][4] = sail.Tensor(self.vit_resampler_attn_mask, [1, 1, 64, 4900], 0)
        self.tensors[self.name_vit]["output"][0] = sail.Tensor(self.vit_out_tensor, [64, self.HIDDEN_SIZE], 0)

        self.tensors[self.name_vit]["input"][0].update_data(np.array(p_pixel_values).reshape(self.tensors[self.name_vit]["input"][0].shape()))
        self.tensors[self.name_vit]["input"][1].update_data(np.array(p_position_ids).reshape(self.tensors[self.name_vit]["input"][1].shape()))
        self.tensors[self.name_vit]["input"][3].update_data(np.array(p_pos_embed_ids).reshape(self.tensors[self.name_vit]["input"][3].shape()))
        
        if full_attn_mask.size == self.MAX_PATCHES * self.MAX_PATCHES:
                self.tensors[self.name_vit]["input"][2].update_data(np.array(full_attn_mask).reshape(self.tensors[self.name_vit]["input"][2].shape()))
                self.tensors[self.name_vit]["input"][4].update_data(np.array(full_attn_mask).reshape(self.tensors[self.name_vit]["input"][4].shape()))
        else:
            mask_full = np.full((self.MAX_PATCHES, self.MAX_PATCHES,), -10000.0, dtype=np.float32)
            mask_full[:hw, :hw] = full_attn_mask
            if hw >= 64:
                mask_full_64 = mask_full[:64, :]
            else:
                mask_full_64 = mask_full[:hw,:]
            self.tensors[self.name_vit]["input"][2].update_data(np.array(mask_full).reshape(self.tensors[self.name_vit]["input"][2].shape()))
            self.tensors[self.name_vit]["input"][4].update_data(np.array(mask_full_64).reshape(self.tensors[self.name_vit]["input"][4].shape()))
        self.model.process(self.name_vit, self.tensors[self.name_vit]["input"], self.tensors[self.name_vit]["output"])
        
        # Concatenate VIT output with text embedding
        vit_out_mem = self.tensors[self.name_vit]["output"][0].asnumpy()
        vit_out_mem = np.expand_dims(vit_out_mem, axis=0) # (1, 64, 2560)
        text_embed_mem = self.tensors[self.name_embed]["output"][0].asnumpy()
        text_embed_mem[:, vit_offset:vit_offset + vit_out_mem.shape[1], :] = vit_out_mem
        self.first_hidden_state[0].update_data(np.array(text_embed_mem).reshape([1, self.SEQLEN, self.HIDDEN_SIZE]))

    def forward_first(self, position_ids):
        position_ids = position_ids[-self.SEQLEN:] if len(position_ids) > self.SEQLEN else position_ids
        self.token_length = len(position_ids)
        length = self.token_length + 1 if self.is_dynamic else self.SEQLEN
        _, _, attention_mask = self.get_first_input(length, position_ids)
        ori_length = len(position_ids)
        position_ids_pad = np.zeros(self.SEQLEN, dtype=np.int32)   # 初始化全 0
        position_ids_pad[:ori_length] = position_ids
        
        # blocks
        for i in range(len(self.dev_ids)):
            self.tensors[self.name_blocks[0]]["input"][3 * i + 1] = sail.Tensor(self.first_pid[i], [1, length], 0)
            self.tensors[self.name_blocks[0]]["input"][3 * i + 1].update_data(position_ids_pad.reshape(self.tensors[self.name_blocks[0]]["input"][3 * i + 1].shape()))
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

        if self.generation_mode == "lmhead_with_topk":
            return_token = int(self.tensors[self.name_lm]["output"][0].asnumpy().item())
        # greedy
        elif self.generation_mode == "greedy":
            self.tensors[self.greedy]["input"][0] = self.tensors[self.name_lm]["output"][0]
            self.model.process(self.greedy, self.tensors[self.greedy]["input"], self.tensors[self.greedy]["output"])
            return_token = int(self.tensors[self.greedy]["output"][0].asnumpy().item())
        else:
            raise ValueError(f"Unsupported generation mode: {self.generation_mode}.")
        
        self.visited_tokens.append(return_token)
        return return_token
    
    def forward_next(self, position_ids):
        self.token_length += 1
        attention_mask = np.zeros(self.SEQLEN+1, self.type_convert(self.tensors[self.name_blocks_cache[0]]["input"][2].dtype()))
        attention_mask[self.token_length - 1:self.SEQLEN] = self.ATTENTION_MASK
        p_ids = position_ids 
            
        # embedding_cache
        if self.generation_mode == "lmhead_with_topk":
            for i in range(len(self.dev_ids)):
                self.tensors[self.name_embed_cache]["input"][i] = self.tensors[self.name_lm]["output"][0]
                self.tensors[self.name_embed_cache]["output"][i] = self.tensors[self.name_blocks_cache[0]]["input"][5 * i]
        elif self.generation_mode == "greedy":
            for i in range(len(self.dev_ids)):
                self.tensors[self.name_embed_cache]["input"][i] = self.tensors[self.greedy]["output"]
                self.tensors[self.name_embed_cache]["output"][i] = self.tensors[self.name_blocks_cache[0]]["input"][5 * i]
        self.model.process(self.name_embed_cache, self.tensors[self.name_embed_cache]["input"], self.tensors[self.name_embed_cache]["output"])

        # block_cache
        for i in range(len(self.dev_ids)):
            self.tensors[self.name_blocks_cache[0]]["input"][5 * i + 1] = self.next_pid[i]
            self.tensors[self.name_blocks_cache[0]]["input"][5 * i + 2] = self.next_attention_mask[i]
            self.tensors[self.name_blocks_cache[0]]["input"][5 * i + 1].update_data(p_ids.reshape(self.tensors[self.name_blocks_cache[0]]["input"][5 * i + 1].shape()))
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
        if self.generation_mode == "lmhead_with_topk":
            return_token = int(self.tensors[self.name_lm]["output"][0].asnumpy().item())
        elif self.generation_mode == "greedy":
            self.tensors[self.greedy]["input"][0] = self.tensors[self.name_lm]["output"][0]
            self.model.process(self.greedy, self.tensors[self.greedy]["input"], self.tensors[self.greedy]["output"])
            return_token = int(self.tensors[self.greedy]["output"][0].asnumpy().item())
        else:
            raise ValueError(f"Unsupported generation mode: {self.generation_mode}.")
        self.visited_tokens.append(return_token)

        return return_token

    def chat(self):
        """
        Start a chat session.
        """
        # Instruct
        print("""\n=================================================================
1. If you want to quit, please enter one of [q, quit, exit]
2. To create a new chat session, please enter one of [clear, new]
=================================================================""")
        # Stop Chatting with "exit" input
        while True:
            self.input_str = input("\nQuestion: ")
            # Quit
            if self.input_str in ["exit", "q", "quit"]:
                break

            # media_path = input("\nImage or Video Path: ")
            media_path = self.input_path

            images = None
            media_path = media_path.strip()
            if media_path == "":
                messages = self.text_message()
                media_type = "text"
            elif not os.path.exists(media_path):
                print("Can't find image or video: {}".format(media_path))
                continue
            else:
                media_type = self.get_media_type(media_path)
                if media_type == "image":
                    images, messages = self.image_message(media_path)
                elif media_type == "video":
                    images, messages = self.video_message(media_path)
                else:
                    print("Unsupported media type: {}".format(media_path))
                    continue
            inputs = self.process_message(images, messages, media_type)
            token_len = inputs.input_ids.numel()
            if token_len >= self.SEQLEN - 128:
                print(
                    "The maximum question length should be shorter than {} but we get {} instead.".
                    format(self.SEQLEN, token_len))
                continue
            print("\nAnswer:")

            # Chat
            first_start = time.time()
            self.forward_embed(inputs.input_ids.squeeze(0).tolist())
            if media_type == "image":
                self.vit_process_image(inputs)
                position_ids = np.arange(inputs.input_ids.shape[1])
                max_posid = int(position_ids.max())
                token = self.forward_first(position_ids)
            elif media_type == "video":
                self.vit_process_video(inputs)
                position_ids = np.arange(inputs.input_ids.shape[1])
                max_posid = int(position_ids.max())
                token = self.forward_first(position_ids)
            else:
                position_ids = np.arange(inputs.input_ids.shape[1])
                max_posid = token_len - 1
                token = self.forward_first(position_ids)
            first_end = time.time()
            tok_num = 0
            # Following tokens
            full_word_tokens = []
            text = ""
            while token not in [self.ID_IMAGE_END, self.ID_END]:
                if self.token_length >= self.SEQLEN:
                    print(f"\ntoken_length has been {self.SEQLEN}, terminated!")
                    break
                full_word_tokens.append(token)
                word = self.tokenizer.decode(full_word_tokens, skip_special_tokens=True)
                if "�" not in word:
                    if len(full_word_tokens) == 1:
                        pre_word = word
                        word = self.tokenizer.decode([token, token],
                                                     skip_special_tokens=True)[len(pre_word):]
                    text += word
                    print(word, flush=True, end="")
                    full_word_tokens = []
                max_posid += 1
                position_ids = np.array([max_posid], dtype=np.int32)
                token = self.forward_next(position_ids)
                tok_num += 1
            next_end = time.time()
            first_duration = first_end - first_start
            next_duration = next_end - first_end
            tps = tok_num / next_duration
            print(f"\nFTL: {first_duration:.3f} s")
            print(f"TPS: {tps:.3f} token/s")


def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    model = MiniCPMV(config["bmodel_path"], config["dev_ids"], config["tokenizer_path"], config["input_path"])
    model.chat()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/minicpmv.yaml', help='path of config file')
    args = parser.parse_args()
    main(args)