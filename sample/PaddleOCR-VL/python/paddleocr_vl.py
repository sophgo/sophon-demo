import argparse
import json
import os
import time

import numpy as np
from PIL import Image
from tokenizers import Tokenizer
from sophon import sail

# fixed image resolution used at bmodel compile time (--max_pixels 784,784)
RESIZED_HEIGHT = 784
RESIZED_WIDTH = 784
PATCH_SIZE = 14


def type_convert(sail_dtype):
    if sail_dtype == sail.Dtype.BM_FLOAT32:
        return np.float32
    if sail_dtype == sail.Dtype.BM_FLOAT16:
        return np.float16
    if sail_dtype == sail.Dtype.BM_INT32:
        return np.int32
    if sail_dtype == sail.Dtype.BM_BFLOAT16:
        return np.uint16
    raise TypeError("only support float32/float16/int32/bfloat16 right now")


def full_mask(shape, mask_value, np_dtype):
    """attention mask filled with the raw mask bit pattern (uint16 for bf16)"""
    arr = np.full(shape, mask_value, dtype=np.uint16)
    if np_dtype == np.float16:
        arr = arr.view(np.float16)
    return arr


def causal_mask(shape, mask_value, np_dtype, token_length):
    """causal mask: mask_value everywhere except j <= i < token_length"""
    arr = np.full(shape, mask_value, dtype=np.uint16)
    if token_length > 0:
        arr[..., :token_length, :token_length] = np.triu(
            np.full((token_length, token_length), mask_value, dtype=np.uint16), 1)
    if np_dtype == np.float16:
        arr = arr.view(np.float16)
    return arr


class PaddleOCR_VL(object):
    def __init__(self, args):
        self.dev_id = args.devid
        self.do_sample_arg = args.do_sample
        print("Device [ %d ] loading ....." % self.dev_id)
        self.handle = sail.Handle(self.dev_id)
        self.net = sail.EngineLLM(args.model_path,
                                  sail.BmrtFlag.BM_RUNTIME_SHARE_MEM,
                                  [self.dev_id])
        print("Model[%s] loading .... Done!" % args.model_path)

        # ===== structure detection, mirrors cpp init_by_names() =====
        self.graph_names = self.net.get_graph_names()
        num_nets = len(self.graph_names)
        num_blocks = num_nets - 4  # embedding, embedding_cache, lm_head, vit
        self.greedy = "greedy_head" if "greedy_head" in self.graph_names else None
        self.sample = "sample_head" if "sample_head" in self.graph_names else None
        if self.greedy is not None:
            num_blocks -= 1
        if self.sample is not None:
            num_blocks -= 1
        self.num_layers = num_blocks // 2
        for i in range(self.num_layers):
            if ("block_%d" % i not in self.graph_names
                    or "block_cache_%d" % i not in self.graph_names):
                self.num_layers = i
                print("Warning: Only %d blocks found, expected %d blocks."
                      % (i, num_blocks // 2))
                break

        self.name_embed = "embedding"
        self.name_embed_cache = "embedding_cache"
        self.name_vit = "vit"
        self.name_lm = "lm_head"
        self.name_blocks = ["block_%d" % i for i in range(self.num_layers)]
        self.name_blocks_cache = ["block_cache_%d" % i for i in range(self.num_layers)]

        embed_cache_out_dtype = self.net.get_output_dtype(self.name_embed_cache, 0)
        if embed_cache_out_dtype == sail.Dtype.BM_FLOAT16:
            self.MASK_VALUE = 0xF0E2  # -65504 by float16
            self.embed_dtype = sail.Dtype.BM_FLOAT16
        elif embed_cache_out_dtype == sail.Dtype.BM_BFLOAT16:
            self.MASK_VALUE = 0xC61C  # -9984 by bfloat16
            self.embed_dtype = sail.Dtype.BM_BFLOAT16
        else:
            raise TypeError("Only support float16/bfloat16 for attention mask")

        self.support_history = self.net.get_input_num(self.name_blocks[0]) == 5
        self.is_dynamic = self.net.get_is_dynamic(self.name_blocks[0])
        self.vit_dynamic = self.net.get_is_dynamic(self.name_vit)
        self.lmhead_with_topk = self.net.get_output_shape(self.name_lm, 0)[1] == 1
        self.MAX_INPUT_LENGTH = self.net.get_input_shape(self.name_embed, 0)[1]
        self.hidden_size = self.net.get_input_shape(self.name_lm, 0)[1]
        self.SEQLEN = self.net.get_input_shape(self.name_blocks_cache[0], 3)[1]
        self.MAX_PATCHES = self.net.get_input_shape(self.name_vit, 0)[0]
        self.PREFILL_KV_LENGTH = 0
        if self.support_history:
            self.PREFILL_KV_LENGTH = self.net.get_input_shape(self.name_blocks[0], 3)[1]
        print("Num Layers: %d" % self.num_layers)
        print("History Support: %s" % self.support_history)

        kv_shape = list(self.net.get_input_shape(self.name_blocks_cache[0], 3))
        self.kv_dtype = self.net.get_input_dtype(self.name_blocks_cache[0], 3)
        # elements of one token's KV cache, e.g. [1,2048,2,128] -> 2*128
        self.kv_elems_per_token = int(np.prod(kv_shape[2:]))
        self.kv_view_shape = [1, 1] + kv_shape[2:]  # [1,1,2,128]

        # ===== tensors =====
        self.input_tensors = {}
        self.output_tensors = {}
        self.input_tensors[self.name_vit] = self.net.create_max_input_tensors(self.name_vit)
        self.output_tensors[self.name_vit] = self.net.create_max_output_tensors(self.name_vit)
        self.input_tensors[self.name_embed] = self.net.create_max_input_tensors(self.name_embed)
        self.output_tensors[self.name_embed] = self.net.create_max_output_tensors(self.name_embed)
        self.input_tensors[self.name_embed_cache] = self.net.create_max_input_tensors(self.name_embed_cache)
        self.output_tensors[self.name_embed_cache] = self.net.create_max_output_tensors(self.name_embed_cache)
        # block_0 template is shared by all prefill blocks (identical shapes)
        self.input_tensors[self.name_blocks[0]] = self.net.create_max_input_tensors(self.name_blocks[0])
        self.output_tensors[self.name_blocks[0]] = self.net.create_max_output_tensors(self.name_blocks[0])

        # decode blocks use the bmodel's internal mems so the KV cache persists
        self.past_key_input = []
        self.past_value_input = []
        for i in range(self.num_layers):
            self.input_tensors[self.name_blocks_cache[i]] = self.net.get_input_tensors(self.name_blocks_cache[i])
            self.output_tensors[self.name_blocks_cache[i]] = self.net.get_output_tensors(self.name_blocks_cache[i])
            self.past_key_input.append(self.input_tensors[self.name_blocks_cache[i]][3])
            self.past_value_input.append(self.input_tensors[self.name_blocks_cache[i]][4])
            self.past_key_input[i].zeros()
            self.past_value_input[i].zeros()

        self.input_tensors[self.name_lm] = self.net.create_max_input_tensors(self.name_lm)
        self.output_tensors[self.name_lm] = self.net.create_max_output_tensors(self.name_lm)
        if self.greedy is not None:
            self.input_tensors[self.greedy] = self.net.create_max_input_tensors(self.greedy)
            self.output_tensors[self.greedy] = self.net.create_max_output_tensors(self.greedy)
        if self.sample is not None:
            self.input_tensors[self.sample] = self.net.create_max_input_tensors(self.sample)
            self.output_tensors[self.sample] = self.net.create_max_output_tensors(self.sample)

        if self.support_history:
            self.dev_buffer = sail.Tensor(self.handle,
                                          [1, self.MAX_INPUT_LENGTH, self.hidden_size],
                                          self.embed_dtype, False, True)
        else:
            self.dev_buffer = None

        # per-layer prefill hidden outputs; sharing one tensor would make each
        # layer read and write the same device mem (cpp uses every net's own
        # stage output_mems, see qwen3_5.py for the same python workaround)
        self.prefill_hidden_outputs = []
        for _ in range(self.num_layers):
            self.prefill_hidden_outputs.append(
                sail.Tensor(self.handle,
                            [1, self.MAX_INPUT_LENGTH, self.hidden_size],
                            self.embed_dtype, False, True))

        # ===== tokenizer =====
        print("Processor [%s] loading .... " % args.config_path, end="")
        self.tok = Tokenizer.from_file(os.path.join(args.config_path, "tokenizer.json"))
        self.ID_VISION_START = self.tok.token_to_id("<|IMAGE_START|>")
        self.ID_EOS = self.tok.token_to_id("</s>")
        # rope pad id, mirrors cpp IMAGE_PAD_TOKEN (<|image_pad|>): note this is
        # NOT <|IMAGE_PLACEHOLDER|> which appears in the prompt; cpp get_rope_index
        # searches this id, never finds it and falls back to sequential positions
        self.ID_IMAGE_PAD = self.tok.token_to_id("<|image_pad|>")
        print("Done!")

        self.spatial_merge_size = 2

        # ===== generation config =====
        self.do_sample = False
        if self.do_sample_arg:
            if self.sample is None:
                print("Warning: Sample head not found in the model. You need compile"
                      " bmodel with --do_sample. Using greedy mode instead!")
            else:
                gen_path = os.path.join(args.config_path, "generation_config.json")
                print("Generation Config [%s] loading .... " % gen_path, end="")
                with open(gen_path) as f:
                    gen_config = json.load(f)
                self.repeat_penalty = gen_config.get("repetition_penalty", 1.0)
                self.temperature = gen_config.get("temperature", 1.0)
                self.top_k = gen_config.get("top_k", 50)
                self.top_p = gen_config.get("top_p", 1.0)
                self.stop_strings = gen_config.get("stop_strings", [])
                self.do_sample = True
                print("Done!")

        # runtime state
        self.token_length = 0
        self.history_length = 0
        self.visited_tokens = np.zeros(self.SEQLEN, dtype=np.int32)
        self.tokens = []
        self.max_posid = 0
        self.history_max_posid = 0
        self.vit_run = False

    # ===== preprocessing (mirrors cpp cv_utils.h) =====
    @staticmethod
    def convert_to_rgb(img):
        if img.mode == "RGBA":
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.getchannel("A"))
            return background
        if img.mode != "RGB":
            return img.convert("RGB")
        return img

    def process_image(self, media_path):
        try:
            img = Image.open(media_path)
        except Exception as e:
            print("Error: Unable to open image file: %s (%s)" % (media_path, e))
            return None, None
        img = self.convert_to_rgb(img)
        img = img.resize((RESIZED_WIDTH, RESIZED_HEIGHT), Image.BICUBIC)
        x = np.asarray(img, dtype=np.float32) / 255.0
        x = (x - 0.5) / 0.5
        # HWC -> CHW flat, same layout as cpp bicubic_resize output
        x = x.transpose(2, 0, 1).reshape(-1)

        grid_h = RESIZED_HEIGHT // PATCH_SIZE  # 56
        grid_w = RESIZED_WIDTH // PATCH_SIZE   # 56
        grid_thw = [1, grid_h, grid_w]
        assert grid_h * grid_w <= self.MAX_PATCHES
        # rearrange_patches: [C,H,Ph,W,Pw] -> [H,W,C,Ph,Pw] -> [patches, 3*14*14]
        x = x.reshape(3, grid_h, PATCH_SIZE, grid_w, PATCH_SIZE)
        x = x.transpose(1, 3, 0, 2, 4).reshape(grid_h * grid_w, 3 * PATCH_SIZE * PATCH_SIZE)
        pixel_values = x.reshape(grid_h * grid_w, 3, PATCH_SIZE, PATCH_SIZE)
        return pixel_values, grid_thw

    # ===== prompt =====
    def build_image_prompt(self, input_str, grid_thws):
        prompt = "<|begin_of_sentence|>User: "
        for (t, h, w) in grid_thws:
            pad_len = h * w // 4
            prompt += "<|IMAGE_START|>"
            prompt += "<|IMAGE_PLACEHOLDER|>" * pad_len
            prompt += "<|IMAGE_END|>"
        prompt += input_str + "\nAssistant: "
        return prompt

    # ===== mrope position ids (mirrors cpp get_rope_index) =====
    def get_rope_index(self, input_ids, grid_thws, pad_id):
        seq_length = len(input_ids)
        vision_start_indices = [i for i, t in enumerate(input_ids) if t == self.ID_VISION_START]
        image_nums = len(vision_start_indices)

        llm_pos_ids_list = []
        st = 0
        remain_images = image_nums
        for img_idx in range(image_nums):
            ed_image = seq_length
            if remain_images > 0:
                for i in range(st, seq_length):
                    if input_ids[i] == pad_id:
                        ed_image = i
                        break
            t, h, w = grid_thws[img_idx]
            remain_images -= 1
            ed = ed_image
            llm_grid_t = t
            llm_grid_h = h // self.spatial_merge_size
            llm_grid_w = w // self.spatial_merge_size
            text_len = ed - st

            st_idx = 0
            if llm_pos_ids_list:
                st_idx = int(np.max(llm_pos_ids_list[-1])) + 1

            text_pos = np.arange(text_len, dtype=np.int64) + st_idx
            llm_pos_ids_list.append(np.stack([text_pos] * 3))

            hw = llm_grid_h * llm_grid_w
            # image token: second_per_grid_t = 0 -> t_index all zeros
            t_index = np.zeros(llm_grid_t * hw, dtype=np.int64)

            h_index = np.tile(np.repeat(np.arange(llm_grid_h), llm_grid_w), llm_grid_t)
            w_index = np.tile(np.arange(llm_grid_w), llm_grid_h * llm_grid_t)
            grid_pos = np.stack([t_index, h_index, w_index]) + (text_len + st_idx)
            llm_pos_ids_list.append(grid_pos)

            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < seq_length:
            st_idx = int(np.max(llm_pos_ids_list[-1])) + 1 if llm_pos_ids_list else 0
            text_len = seq_length - st
            text_pos = np.arange(text_len, dtype=np.int64) + st_idx
            llm_pos_ids_list.append(np.stack([text_pos] * 3))

        return np.concatenate(llm_pos_ids_list, axis=1).astype(np.int32)  # [3, seq_len]

    # ===== forward passes (mirror cpp chat.cpp) =====
    def forward_embed(self, tokens):
        self.visited_tokens[:] = 0
        self.visited_tokens[:len(tokens)] = tokens
        embed_in_dtype = type_convert(self.input_tensors[self.name_embed][0].dtype())
        input_ids = np.zeros((1, self.MAX_INPUT_LENGTH), dtype=embed_in_dtype)
        input_ids[0, :len(tokens)] = tokens
        self.input_tensors[self.name_embed][0].update_data(input_ids)
        self.net.process(self.name_embed, self.input_tensors[self.name_embed],
                         self.output_tensors[self.name_embed])
        if self.support_history:
            self.dev_buffer.zeros()
            self.dev_buffer.sync_d2d(self.output_tensors[self.name_embed][0], 0, 0,
                                     len(tokens) * self.hidden_size)
        self.token_length = len(tokens)
        self.tokens = list(tokens)

    def forward_vit(self, pixel_values, grid_thw, vit_offset):
        t, h, w = grid_thw
        hw = t * h * w
        if self.vit_dynamic:
            self.input_tensors[self.name_vit][0].reshape(list(pixel_values.shape))
            self.input_tensors[self.name_vit][0].update_data(
                pixel_values.astype(type_convert(self.input_tensors[self.name_vit][0].dtype())))
        else:
            vit_shape = list(self.input_tensors[self.name_vit][0].shape())
            pixel_values_full = np.zeros(vit_shape, dtype=np.float32)
            pixel_values_full[:hw] = pixel_values
            self.input_tensors[self.name_vit][0].update_data(pixel_values_full)
        self.net.process(self.name_vit, self.input_tensors[self.name_vit],
                         self.output_tensors[self.name_vit])
        vit_size = (hw // 4) * self.hidden_size
        embed_target = self.dev_buffer if self.support_history else self.output_tensors[self.name_embed][0]
        embed_target.sync_d2d(self.output_tensors[self.name_vit][0], 0,
                              vit_offset * self.hidden_size, vit_size)
        self.vit_run = True

    def generate(self):
        if self.lmhead_with_topk:
            return int(self.output_tensors[self.name_lm][0].asnumpy().item())
        if self.do_sample:
            return self.penalty_sample()
        return self.greedy_search()

    def greedy_search(self):
        self.input_tensors[self.greedy][0] = self.output_tensors[self.name_lm][0]
        self.net.process(self.greedy, self.input_tensors[self.greedy],
                         self.output_tensors[self.greedy])
        return int(self.output_tensors[self.greedy][0].asnumpy())

    def penalty_sample(self):
        self.input_tensors[self.sample][0] = self.output_tensors[self.name_lm][0]
        generated_tokens = np.zeros(
            [1, self.MAX_INPUT_LENGTH],
            type_convert(self.input_tensors[self.sample][1].dtype()))
        n_tok = min(self.token_length, self.MAX_INPUT_LENGTH)
        generated_tokens[0, :n_tok] = self.visited_tokens[:n_tok]
        self.input_tensors[self.sample][1].update_data(generated_tokens)
        self.input_tensors[self.sample][2].update_data([self.repeat_penalty])
        self.input_tensors[self.sample][3].update_data([self.temperature])
        top_k_arr = np.ones([1], type_convert(self.input_tensors[self.sample][4].dtype())) * self.top_k
        self.input_tensors[self.sample][4].update_data(top_k_arr)
        self.input_tensors[self.sample][5].update_data([self.top_p])
        self.net.process(self.sample, self.input_tensors[self.sample],
                         self.output_tensors[self.sample])
        probs = self.output_tensors[self.sample][0].asnumpy()[0, :self.top_k]
        token_TopK = self.output_tensors[self.sample][1].asnumpy()[0, :self.top_k]
        return int(np.random.choice(token_TopK, p=probs / probs.sum()))

    def forward_first(self, position_ids_1d):
        if self.support_history:
            return self.forward_first_with_kv(position_ids_1d)

        pos_dtype = type_convert(self.input_tensors[self.name_blocks[0]][1].dtype())
        mask_dtype = type_convert(self.input_tensors[self.name_blocks[0]][2].dtype())
        if self.is_dynamic:
            length = self.token_length
            position_ids_pad = np.asarray(position_ids_1d, dtype=pos_dtype).reshape(3, length)
            attention_mask = causal_mask((length, length), self.MASK_VALUE, mask_dtype, length)
            attention_mask = attention_mask.reshape(1, 1, length, length)
        else:
            length = self.MAX_INPUT_LENGTH
            ori_length = len(position_ids_1d) // 3
            position_ids_pad = np.zeros((3, length), dtype=pos_dtype)
            position_ids_pad[:, :ori_length] = \
                np.asarray(position_ids_1d, dtype=pos_dtype).reshape(3, ori_length)
            attention_mask = causal_mask((length, length), self.MASK_VALUE, mask_dtype,
                                         self.token_length).reshape(1, 1, length, length)

        block_inputs = self.input_tensors[self.name_blocks[0]]
        if self.is_dynamic:
            block_inputs[1].reshape([3, length])
            block_inputs[2].reshape([1, 1, length, length])
        block_inputs[1].update_data(position_ids_pad)
        block_inputs[2].update_data(attention_mask)

        out_mem = self.output_tensors[self.name_embed][0]
        kv_elems = self.kv_elems_per_token * self.token_length
        for idx in range(self.num_layers):
            if self.is_dynamic:
                block_inputs[0] = sail.Tensor(out_mem, [1, length, self.hidden_size], 0)
            else:
                block_inputs[0] = out_mem
            block_outputs = {
                0: self.prefill_hidden_outputs[idx],
                1: self.output_tensors[self.name_blocks[0]][1],
                2: self.output_tensors[self.name_blocks[0]][2],
            }
            self.net.process(self.name_blocks[idx], block_inputs, block_outputs)
            self.past_key_input[idx].sync_d2d(block_outputs[1], 0, 0, kv_elems)
            self.past_value_input[idx].sync_d2d(block_outputs[2], 0, 0, kv_elems)
            out_mem = self.prefill_hidden_outputs[idx]
        self.vit_run = False

        lm_input = sail.Tensor(out_mem, [1, self.hidden_size],
                               (self.token_length - 1) * self.hidden_size)
        self.net.process(self.name_lm, {0: lm_input}, self.output_tensors[self.name_lm])
        token = self.generate()
        self.visited_tokens[self.token_length] = token
        self.tokens.append(token)
        self.token_length += 1
        self.history_length = self.token_length
        return token

    def forward_first_with_kv(self, position_ids_1d):
        ori_length = len(position_ids_1d) // 3
        assert ori_length == self.token_length
        old_length = self.history_length
        self.history_length += self.token_length
        assert self.history_length < self.SEQLEN
        assert old_length <= self.PREFILL_KV_LENGTH
        max_kv_length = self.MAX_INPUT_LENGTH + self.PREFILL_KV_LENGTH

        pos_dtype = type_convert(self.input_tensors[self.name_blocks[0]][1].dtype())
        mask_dtype = type_convert(self.input_tensors[self.name_blocks[0]][2].dtype())
        attention_mask = full_mask((1, 1, self.MAX_INPUT_LENGTH, max_kv_length),
                                   self.MASK_VALUE, mask_dtype)
        attention_mask[0, 0, :self.token_length, :old_length] = 0
        tl = self.token_length
        attention_mask[0, 0, :tl, self.PREFILL_KV_LENGTH:self.PREFILL_KV_LENGTH + tl] = np.triu(
            np.full((tl, tl), self.MASK_VALUE, dtype=np.uint16), 1)

        position_ids_pad = np.zeros((3, self.MAX_INPUT_LENGTH), dtype=pos_dtype)
        position_ids_pad[:, :ori_length] = \
            np.asarray(position_ids_1d, dtype=pos_dtype).reshape(3, ori_length)

        block_inputs = self.input_tensors[self.name_blocks[0]]
        block_inputs[1].update_data(position_ids_pad)
        block_inputs[2].update_data(attention_mask)

        out_mem = self.dev_buffer
        old_kv_elems = self.kv_elems_per_token * old_length
        new_kv_elems = self.kv_elems_per_token * self.token_length
        for idx in range(self.num_layers):
            block_inputs[0] = out_mem
            if old_length > 0:
                block_inputs[3].sync_d2d(self.past_key_input[idx], 0, 0, old_kv_elems)
                block_inputs[4].sync_d2d(self.past_value_input[idx], 0, 0, old_kv_elems)
            else:
                block_inputs[3].zeros()
                block_inputs[4].zeros()
            block_outputs = {
                0: self.prefill_hidden_outputs[idx],
                1: self.output_tensors[self.name_blocks[0]][1],
                2: self.output_tensors[self.name_blocks[0]][2],
            }
            self.net.process(self.name_blocks[idx], block_inputs, block_outputs)
            self.past_key_input[idx].sync_d2d(block_outputs[1], 0,
                                              old_length * self.kv_elems_per_token, new_kv_elems)
            self.past_value_input[idx].sync_d2d(block_outputs[2], 0,
                                                old_length * self.kv_elems_per_token, new_kv_elems)
            out_mem = self.prefill_hidden_outputs[idx]
        self.vit_run = False

        lm_input = sail.Tensor(out_mem, [1, self.hidden_size],
                               (self.token_length - 1) * self.hidden_size)
        self.net.process(self.name_lm, {0: lm_input}, self.output_tensors[self.name_lm])
        token = self.generate()
        self.visited_tokens[self.token_length] = token
        self.tokens.append(token)
        self.token_length += 1
        self.history_length += 1
        return token

    def forward_next(self, position_ids):
        mask_shape = list(self.net.get_input_shape(self.name_blocks_cache[0], 2))  # [1,1,1,SEQLEN+1]
        mask_dtype = type_convert(self.net.get_input_dtype(self.name_blocks_cache[0], 2))
        attention_mask = np.zeros(mask_shape, dtype=np.uint16)
        # mask [history_length-1, SEQLEN); the last (SEQLEN+1-th) slot holds the
        # current token's own K/V inside the graph and must stay unmasked
        attention_mask[0, 0, 0, self.history_length - 1:self.SEQLEN] = self.MASK_VALUE
        if mask_dtype == np.float16:
            attention_mask = attention_mask.view(np.float16)

        token_input = np.array([[self.visited_tokens[self.token_length - 1]]],
                               dtype=type_convert(self.input_tensors[self.name_embed_cache][0].dtype()))
        self.input_tensors[self.name_embed_cache][0].update_data(token_input)
        self.net.process(self.name_embed_cache, self.input_tensors[self.name_embed_cache],
                         self.output_tensors[self.name_embed_cache])
        out_mem = self.output_tensors[self.name_embed_cache][0]

        # pos/mask updated once on block_cache_0's tensors, shared by all layers
        bc0_inputs = self.input_tensors[self.name_blocks_cache[0]]
        bc0_inputs[1].update_data(np.asarray(position_ids, dtype=np.int32).reshape(3, 1))
        bc0_inputs[2].update_data(attention_mask)

        token_offset = (self.history_length - 1) * self.kv_elems_per_token
        if not hasattr(self, "_kv_view_keepalive"):
            self._kv_view_keepalive = []
        for idx in range(self.num_layers):
            new_k = sail.Tensor(self.past_key_input[idx], self.kv_view_shape, token_offset)
            new_v = sail.Tensor(self.past_value_input[idx], self.kv_view_shape, token_offset)
            # the view tensors must outlive the launch: sail frees the
            # underlying device mem when the last reference drops
            self._kv_view_keepalive.append((new_k, new_v))
            block_inputs = {
                0: out_mem,
                1: bc0_inputs[1],
                2: bc0_inputs[2],
                3: self.past_key_input[idx],
                4: self.past_value_input[idx],
            }
            block_outputs = {
                0: self.output_tensors[self.name_blocks_cache[idx]][0],
                1: new_k,
                2: new_v,
            }
            self.net.process(self.name_blocks_cache[idx], block_inputs, block_outputs)
            # d2s fence: without a readback after the block launch the KV
            # cache write races with the next layer's launch on SoC and the
            # decode output becomes non-deterministic run to run; a >=4KB
            # readback per layer keeps it deterministic (verified against the
            # C++ sample token-by-token). Costs ~17ms/token on BM1688.
            _ = self.past_key_input[idx].asnumpy()[0, 0, 0, :2048]
            out_mem = self.output_tensors[self.name_blocks_cache[idx]][0]

        self.net.process(self.name_lm, {0: out_mem}, self.output_tensors[self.name_lm])
        token = self.generate()
        self.visited_tokens[self.token_length] = token
        self.tokens.append(token)
        self.token_length += 1
        self.history_length += 1
        return token

    def clear_history(self):
        if not self.support_history:
            return
        for i in range(self.num_layers):
            self.past_key_input[i].zeros()
            self.past_value_input[i].zeros()
        self.history_length = 0

    def forward_prefill(self, position_ids_1d):
        if self.history_length == 0 or not self.support_history:
            self.history_max_posid = 0
            return self.forward_first(position_ids_1d)
        if (self.history_length + self.token_length + 128 > self.SEQLEN
                or self.history_length > self.PREFILL_KV_LENGTH):
            print("Warning: History is full and clear it to continue.")
            self.clear_history()
            self.history_max_posid = 0
            return self.forward_first(position_ids_1d)
        # all ids should increase by history_max_posid
        position_ids_1d = [x + self.history_max_posid for x in position_ids_1d]
        self.max_posid += self.history_max_posid
        return self.forward_first(position_ids_1d)

    def check_stop(self, text):
        for stop_str in self.stop_strings:
            if text.endswith(stop_str):
                return True
        return False

    # ===== chat loop (mirrors cpp pipeline.cpp) =====
    def chat(self):
        mode_map = {
            "ocr": "OCR:",
            "table": "Table Recognition:",
            "formula": "Formula Recognition:",
            "chart": "Chart Recognition:",
        }
        print("\n=================================================================\n"
              "1. If you want to quit, please enter one of [q, quit, exit]\n"
              "2. To create a new chat session, please enter one of [clear, new]\n"
              "=================================================================")
        while True:
            try:
                input_str = input("\nMode(ocr,table,formula,chart): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExit.")
                break
            if input_str in ("exit", "q", "quit"):
                break
            if input_str in ("clear", "c", "new"):
                self.clear_history()
                self.history_max_posid = 0
                print("Chat history cleared.")
                continue
            if input_str not in mode_map:
                print("Unsupported mode. Please enter one of the following modes:"
                      " ocr, table, formula, chart.")
                continue
            mode_str = mode_map[input_str]

            try:
                media_line = input("\nImage Path: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExit.")
                break
            if not media_line:
                media_line = "../image.png"
            medias = [m.strip() for m in media_line.split(",") if m.strip()]
            missing = [m for m in medias if not os.path.exists(m)]
            if missing:
                print("File does not exist: %s" % ", ".join(missing))
                continue

            print("\nAnswer:")
            pixel_values_list = []
            grid_thws = []
            for m in medias:
                pixel_values, grid_thw = self.process_image(m)
                if pixel_values is None:
                    print("Error processing image: %s" % m)
                    pixel_values_list = []
                    break
                pixel_values_list.append(pixel_values)
                grid_thws.append(grid_thw)
            if not pixel_values_list:
                continue

            sentence_input = self.build_image_prompt(mode_str, grid_thws)
            tokens = self.tok.encode(sentence_input).ids
            if len(tokens) > self.MAX_INPUT_LENGTH:
                print("Input tokens exceed maximum length: %d" % self.MAX_INPUT_LENGTH)
                continue
            input_token_num = len(tokens)
            vit_offsets = [i for i, t in enumerate(tokens) if t == self.ID_VISION_START]

            clock_start = time.time()
            self.forward_embed(tokens)
            clock_vit_start = time.time()
            for pixel_values, grid_thw, vit_offset in zip(pixel_values_list, grid_thws, vit_offsets):
                self.forward_vit(pixel_values, grid_thw, vit_offset + 1)
            duration_vit = (time.time() - clock_vit_start) * 1000

            position_ids = self.get_rope_index(tokens, grid_thws, self.ID_IMAGE_PAD)
            self.max_posid = int(position_ids[0].max())
            token = self.forward_prefill(position_ids.flatten().tolist())
            clock_prefill = time.time()
            duration_prefill = (clock_prefill - clock_start) * 1000

            full_word_tokens = []
            text = ""
            output_token_num = 0
            while token != self.ID_EOS and self.history_length < self.SEQLEN:
                full_word_tokens.append(token)
                word = self.tok.decode(full_word_tokens)
                if "�" not in word:
                    if len(full_word_tokens) == 1:
                        pre_word = word
                        word = self.tok.decode([token, token])[len(pre_word):]
                    text += word
                    print(word, end="", flush=True)
                    if self.do_sample and self.check_stop(text):
                        break
                    full_word_tokens = []
                self.max_posid += 1
                token = self.forward_next([self.max_posid] * 3)
                output_token_num += 1
            self.history_max_posid = self.max_posid + 2

            print()
            duration_decode = (time.time() - clock_prefill) * 1000
            print("FTL: %.3f s" % (duration_prefill / 1000.0))
            if output_token_num > 0:
                print("TPS: %.4f tokens/s" % (output_token_num * 1000.0 / duration_decode))
            if duration_vit > 0:
                t, h, w = grid_thws[-1]
                print("Vision [%d, %d, %d]: %.3f s" % (t, h, w, duration_vit / 1000.0))
            print("Input Tokens: %d, Output Tokens: %d" % (input_token_num, output_token_num + 1))


def main(args):
    model = PaddleOCR_VL(args)
    model.chat()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str, required=True,
                        help="path to the bmodel file")
    parser.add_argument("-c", "--config_path", type=str, default="../config",
                        help="path to the config directory")
    parser.add_argument("-d", "--devid", type=int, default=0,
                        help="device ID to use, default is 0")
    parser.add_argument("-s", "--do_sample", action="store_true",
                        help="if set, generate tokens by sample parameters")
    args = parser.parse_args()
    main(args)
