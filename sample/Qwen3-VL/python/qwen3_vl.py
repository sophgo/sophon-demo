import time
import os
import argparse
from sophon import sail
from transformers import AutoProcessor, GenerationConfig
from qwen_vl_utils import process_vision_info
import json
import torch
import numpy as np
import torch.nn.functional as F
import SILK2.Tools.logger as Logger

# Preprocess the images
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

#convert sail_dtype to numpy dtype
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



class Qwen3_VL():

    def __init__(self, args):
        self.version = "1.0.0"
        # init logger
        self.logger = Logger.Log("Qwen3VL", args.log_level)
        self.logger.info(f"{Logger.file_lineno()} loading model {args.model_path} to dev:{args.devid}")
        st = time.time()
        self.media_path = None # 初始化媒体路径缓存
        # devid        
        self.dev_id = args.devid
        self.video_ratio = args.video_ratio

        self.handle = sail.Handle(self.dev_id)
        # LLM bmodel inference on A2 may cause device memory increasing, which should be solved by set bmrt_set_flags(BM_RUNTIME_SHARE_MEM).
        self.net = sail.EngineLLM(args.model_path, sail.BmrtFlag.BM_RUNTIME_SHARE_MEM, [self.dev_id])
        self.logger.info(f"{Logger.file_lineno()} load model cost: {time.time() - st}")
        self.logger.info(f"{Logger.file_lineno()} model loaded!")
        self.logger.info(f"{Logger.file_lineno()} init input/output tensors")
        st = time.time()
        # graph
        self.graph_names = self.net.get_graph_names()
        # initialize qwen parameters
        self.num_layers = 0
        for graph_name in self.graph_names:
            if "block_cache_" in graph_name:
                self.num_layers += 1

        self.vit_hidden_states_input_shape = self.net.get_input_shape("vit", 0)
        self.vit_pos_ids_input_shape = self.net.get_input_shape("vit", 1)
        self.pos_idx_input_shape = self.net.get_input_shape("vit", 2)
        self.pos_weight_input_shape = self.net.get_input_shape("vit", 3)
        self.vit_attention_mask_input_shape = self.net.get_input_shape("vit", 4)

        self.seq_len = self.net.get_input_shape("block_cache_0", 3)[1]
        self.vision_seq_len = self.vit_hidden_states_input_shape[0]
        self.hidden_size = self.net.get_input_shape("lm_head", 0)[1]
        self.input_tensors = {}
        self.output_tensors = {}
        self.past_kv_stride = [1] * len(self.net.get_input_shape("block_cache_0", 3))
        for dim_i in range(len(self.net.get_input_shape("block_cache_0", 3))-2, -1, -1):
            self.past_kv_stride[dim_i] = self.net.get_input_shape("block_cache_0", 3)[dim_i + 1] * \
                                            self.past_kv_stride[dim_i + 1]
        self.vision_seq_max_ratio = 0.8 # apply auto resize
        self.tokens = []
        self.do_sample = args.do_sample
        self.is_dynamic = self.net.get_is_dynamic("block_0")
        self.is_vit_dynamic = self.net.get_is_dynamic("vit")
        self.is_add_dynamic = self.net.get_is_dynamic("add")
        self.MAX_PATCHES = self.vit_hidden_states_input_shape[0]
        self.MAX_PIXELS = self.MAX_PATCHES * 16 * 16
        self.support_history = False
        self.MAX_INPUT_LENGTH = self.net.get_input_shape("embedding", 0)[1]
        self.PREFILL_KV_LENGTH = self.net.get_input_shape("block_cache_0", 3)[1]
        
        # initialize net name
        self.name_embed = "embedding"
        self.name_embed_cache = "embedding_cache"
        self.name_lm = "lm_head"
        self.name_blocks = ["block_"+str(i) for i in range(self.num_layers)]
        self.name_blocks_cache = ["block_cache_"+str(i) for i in range(self.num_layers)]
        self.name_vit = "vit"
        self.greedy = "greedy_head"
        self.sample = "sample_head"
        self.name_add = "add"
        self.name_deepstack = "deepstack"

        if args.do_sample and self.sample in self.graph_names:
            self.generation_mode = "sample"
            self.logger.info(f"{Logger.file_lineno()} Generation mode: sample")
        elif self.greedy in self.graph_names:
            self.generation_mode = "greedy"
            self.logger.info(f"{Logger.file_lineno()} Generation mode: greedy")
        else:
            self.generation_mode = None
            self.logger.info(f"{Logger.file_lineno()} Generation mode: lmhead_with_topk")

        # initialize vision tensors (inputs & outputs)
        self.input_tensors[self.name_vit] = self.net.create_max_input_tensors(self.name_vit)
        self.output_tensors[self.name_vit] = self.net.create_max_output_tensors(self.name_vit)

        # forward_first: embedding tensors (inputs & outputs)
        self.input_tensors[self.name_embed] = self.net.create_max_input_tensors(self.name_embed)
        self.output_tensors[self.name_embed] = self.net.create_max_output_tensors(self.name_embed)

        # forward_next: embedding tensors (inputs & outputs)
        self.input_tensors[self.name_embed_cache] = self.net.create_max_input_tensors(self.name_embed_cache)
        self.output_tensors[self.name_embed_cache] = self.net.create_max_output_tensors(self.name_embed_cache)

        # forward_first: hidden_state, position_id_tensor and attention_mask tensors (inputs & outputs)
        self.input_tensors[self.name_blocks[0]] = self.net.create_max_input_tensors(self.name_blocks[0])
        self.first_hidden_states_output = self.init_sail_tensor(self.name_blocks[0], 0, is_input=False)

        # forward_next: hidden_state, position_id_tensor and attention_mask tensors (inputs & outputs)
        self.next_hidden_states_input = self.init_sail_tensor(self.name_blocks_cache[0], 0)
        self.next_pos_ids_input = self.init_sail_tensor(self.name_blocks_cache[0], 1)
        self.next_attention_mask_input = self.init_sail_tensor(self.name_blocks_cache[0], 2)
        self.next_hidden_states_output = self.init_sail_tensor(self.name_blocks_cache[0], 0, is_input=False)

        # forward_next/forward_first: present_key / present_value (for update kv_cache)
        self.present_key_output = self.init_sail_tensor(self.name_blocks_cache[0], 1, is_input=False)
        self.present_value_output = self.init_sail_tensor(self.name_blocks_cache[0], 2, is_input=False)

        # forward_first: key_tensor and value_tensor
        self.past_key_input = []
        self.past_value_input = []
        for _ in range(self.num_layers): 
            self.past_key_input.append(self.init_sail_tensor(self.name_blocks_cache[0], 3))
            self.past_value_input.append(self.init_sail_tensor(self.name_blocks_cache[0], 4))

        # lm_head tensors (inputs & outputs)
        self.input_tensors[self.name_lm] = self.net.create_max_input_tensors(self.name_lm)
        self.output_tensors[self.name_lm] = self.net.create_max_output_tensors(self.name_lm)

        # add tensors (inputs & outputs)
        self.input_tensors[self.name_add] = self.net.create_max_input_tensors(self.name_add)
        self.output_tensors[self.name_add] = self.net.create_max_output_tensors(self.name_add)

        # sample tensors (inputs & outputs)
        if self.generation_mode is not None:
            self.input_tensors[self.greedy] = self.net.create_max_input_tensors(self.greedy)
            self.output_tensors[self.greedy] = self.net.create_max_output_tensors(self.greedy)
            self.input_tensors[self.sample] = self.net.create_max_input_tensors(self.sample)
            self.output_tensors[self.sample] = self.net.create_max_output_tensors(self.sample)

        self.logger.info(f"{Logger.file_lineno()} tensor init cost: {time.time() - st}")
        self.logger.info(f"{Logger.file_lineno()} init input/output tensors finish!")

        self.logger.info(f"{Logger.file_lineno()} init tokenizer and preprocessor")
        st = time.time()

        # init preprocessor & tokenizer & configs
        config_dir = args.config_path
        self.processor = AutoProcessor.from_pretrained(config_dir, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        self.ID_END = self.tokenizer.convert_tokens_to_ids("<|end|>")
        self.ID_IM_END = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.ID_IMAGE_PAD = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.ID_VIDEO_PAD = self.tokenizer.convert_tokens_to_ids("<|video_pad|>")
        self.ID_VISION_START = self.tokenizer.convert_tokens_to_ids('<|vision_start|>')
        self.spatial_merge_size = 2
        self.spatial_merge_unit = self.spatial_merge_size**2
        self.tokens_per_second = 2
        self.support_history = self.support_history
        self.num_grid_per_side = 48
        self.max_posid = 0
        self.history_max_posid = 0
        self.total_pixels = (self.MAX_INPUT_LENGTH - 128) * 32 * 32
        self.num_deepstack = 3
        self.deepstack_buffers = []
        self.vit_run =False
        for i in range(self.num_deepstack):
            self.deepstack_buffers.append(sail.Tensor(self.handle, [1,self.MAX_INPUT_LENGTH,self.hidden_size], sail.Dtype.BM_BFLOAT16, False, True))

        with open(str(config_dir + '/config.json'), 'r') as f:
            self.config = json.load(f)
        self.token_len = 0
        self.logger.debug(f"{Logger.file_lineno()} end token ids: {self.ID_IM_END}/{self.ID_END}, max step: {self.seq_len}")

        if self.generation_mode == "sample":
            gen_config = GenerationConfig.from_pretrained(config_dir)
            self.temperature = getattr(gen_config, "temperature", 0.8)
            self.top_p = getattr(gen_config, "top_p", 0.8)
            self.top_k = getattr(gen_config, "top_k", 50)
            self.repeat_penalty = getattr(gen_config, "repeat_penalty", 1.1)

        # init runtime val
        self.init_runtime_vals()
        self.logger.info(f"{Logger.file_lineno()} init tokenizer and preprocessor cost: {time.time() - st}")
        self.logger.info(f"{Logger.file_lineno()} init tokenizer and preprocessor finish!")

    def get_dev_id(self):
        return self.dev_id

    def init_runtime_vals(self):
        self.step = 0
        self.token_pos_length = 0
        self.last_id = None
        self.logger.debug(f"{Logger.file_lineno()} clear runtime vals success!")
    

    def init_sail_tensor(self, name, tensor_idx, shape=None, is_input=True):
        """
        init a sail tensor of sail.engine.
        parameters:
        input:
            name: str, graph_name/net_name
            tensor_idx: int, input/output tensor id
            shape: list[int], shape of tensor
            is_input: bool, is input tensor or not
        return:
            sail.Tensor
        """
        tensor = {}
        if is_input:
            tensor["name"] = self.net.get_input_names(name)[tensor_idx]
            tensor["shape"] = self.net.get_input_shape(name, tensor_idx) if shape is None else shape
            tensor["dtype"] = self.net.get_input_dtype(name, tensor_idx)
            tensor["data"] = sail.Tensor(self.handle, tensor["shape"], tensor["dtype"], False, True)
        else:
            tensor["name"] = self.net.get_output_names(name)[tensor_idx]
            tensor["shape"] = self.net.get_output_shape(name, tensor_idx) if shape is None else shape
            tensor["dtype"] = self.net.get_output_dtype(name, tensor_idx)
            tensor["data"] = sail.Tensor(self.handle, tensor["shape"], tensor["dtype"], False, True) 
        return tensor["data"]

    def text_message(self):
        # yapf: disable
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": self.input_str}],
        }]
        # yapf: enable
        return messages

    def image_message(self, path):
        # yapf: disable
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": path,
                 "min_pixels": 4 * 32 * 32,
                 "max_pixels": self.MAX_PIXELS},
                {"type": "text", "text": self.input_str},
            ],
        }]
        # yapf: enable
        return messages

    def video_message(self, path):
        # yapf: disable
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": path, "fps": 1.0,
                 "min_pixels": 4 * 32 * 32,
                 "max_pixels": int(self.MAX_PIXELS * self.video_ratio),
                 "total_pixels": self.total_pixels},
                {"type": "text", "text": self.input_str},
            ],
        }]
        # yapf: enable
        return messages
    
    def clear_history(self):
        self.past_key_input = []
        self.past_value_input = []
        self.step = 0
        for _ in range(self.num_layers): 
            self.past_key_input.append(self.init_sail_tensor(self.name_blocks_cache[0], 3))
            self.past_value_input.append(self.init_sail_tensor(self.name_blocks_cache[0], 4))

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

    def rot_pos(self, grid_thw):
        merge_size = self.spatial_merge_size
        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.int32)

        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h, merged_w = height // merge_size, width // merge_size

            block_rows = torch.arange(merged_h)  # block row indices
            block_cols = torch.arange(merged_w)  # block col indices
            intra_row = torch.arange(merge_size)  # intra-block row offsets
            intra_col = torch.arange(merge_size)  # intra-block col offsets

            # Compute full-resolution positions
            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)

            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset:offset + num_tokens] = coords
            offset += num_tokens

            # lookup rotary embeddings
        return pos_ids
    
    def fast_pos_embed_interpolate(self, grid_thw):
        t, h, w = grid_thw[0]
        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]
        h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
        w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

        h_idxs_floor = h_idxs.int()
        w_idxs_floor = w_idxs.int()
        h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
        w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

        dh = h_idxs - h_idxs_floor
        dw = w_idxs - w_idxs_floor

        base_h = h_idxs_floor * self.num_grid_per_side
        base_h_ceil = h_idxs_ceil * self.num_grid_per_side

        indices = [
            (base_h[None].T + w_idxs_floor[None]).flatten(),
            (base_h[None].T + w_idxs_ceil[None]).flatten(),
            (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
            (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
        ]

        weights = [
            ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
            ((1 - dh)[None].T * dw[None]).flatten(),
            (dh[None].T * (1 - dw)[None]).flatten(),
            (dh[None].T * dw[None]).flatten(),
        ]

        for i in range(4):
            idx_list[i].extend(indices[i].tolist())
            weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.int32)
        weight_tensor = torch.tensor(weight_list, dtype=torch.float32)
        msize = self.spatial_merge_size
        idx_tensor = idx_tensor.view(4, t, h // msize, msize, w // msize,
                                     msize).permute(1, 2, 4, 3, 5, 0).reshape(t * h * w, 4)
        weight_tensor = weight_tensor.view(4, t, h // msize, msize, w // msize,
                                           msize).permute(1, 2, 4, 3, 5, 0).reshape(t * h * w, 4)

        return idx_tensor, weight_tensor

    def vit_process_image(self, inputs):
        vit_token_list = torch.where(inputs.input_ids == self.ID_VISION_START)[1].tolist()
        pre_patches = 0
        for idx, vit_offset in enumerate(vit_token_list):
            grid_thw = inputs.image_grid_thw[idx].unsqueeze(0)
            num_patches = int(torch.prod(grid_thw))
            hidden_states = inputs.pixel_values[pre_patches:pre_patches + num_patches, :]
            position_ids = self.rot_pos(grid_thw)
            pos_ids, pos_weights = self.fast_pos_embed_interpolate(grid_thw.tolist())
            self.forward_vit(hidden_states.numpy(), position_ids.numpy(), pos_ids.numpy(),
                                   pos_weights.numpy(), grid_thw.numpy(), vit_offset + 1)
            pre_patches += num_patches

    def vit_process_video(self, inputs):
        vit_token_list = torch.where(inputs.input_ids == self.ID_VISION_START)[1].tolist()
        t, h, w = inputs.video_grid_thw.flatten().tolist()
        assert (t == len(vit_token_list))
        grid_thw = torch.tensor([[1, h, w]], dtype=torch.int32)
        position_ids = self.rot_pos(grid_thw)
        pos_ids, pos_weights = self.fast_pos_embed_interpolate(grid_thw.tolist())
        for idx, vit_offset in enumerate(vit_token_list):
            hidden_states = inputs.pixel_values_videos[(idx * h * w):((idx + 1) * h * w), :]
            self.forward_vit(hidden_states.numpy(), position_ids.numpy(), pos_ids.numpy(),
                                   pos_weights.numpy(), grid_thw.numpy(), vit_offset + 1)

    def get_rope_index(self, input_ids: torch.LongTensor, grid_thw: torch.LongTensor,
                       pad_id: int) -> torch.Tensor:
        total_input_ids = input_ids
        position_ids = torch.ones(3, input_ids.shape[0], input_ids.shape[1])
        image_index = 0
        for i, input_ids in enumerate(total_input_ids):
            vision_start_indices = torch.argwhere(input_ids == self.ID_VISION_START).squeeze(1)
            image_nums = len(vision_start_indices)
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images = image_nums
            for _ in range(image_nums):
                if pad_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(pad_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if pad_id == self.ID_IMAGE_PAD:
                    t, h, w = (
                        grid_thw[image_index][0].item(),
                        grid_thw[image_index][1].item(),
                        grid_thw[image_index][2].item(),
                    )
                else:
                    t, h, w = 1, grid_thw[0][1].item(), grid_thw[0][2].item()
                image_index += 1
                remain_images -= 1
                ed = ed_image

                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t,
                    h // self.spatial_merge_size,
                    w // self.spatial_merge_size,
                )
                text_len = ed - st

                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                # t_index is always 0 because llm_grid_t is always 1 (we use timestamps to encode the temporal information for videos)
                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h *
                                                                      llm_grid_w).flatten()

                h_index = torch.arange(llm_grid_h).view(1, -1,
                                                        1).expand(llm_grid_t, -1,
                                                                  llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1,
                                                        -1).expand(llm_grid_t, llm_grid_h,
                                                                   -1).flatten()
                llm_pos_ids_list.append(
                    torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, :] = llm_positions
        return position_ids.to(torch.int32)
        
    
    def get_attn_mask(self, seq_length, cu_seqlens):
        attention_mask = torch.full([1, 1, seq_length, seq_length], -10000.0, dtype=torch.float32)
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1]:cu_seqlens[i],
                           cu_seqlens[i - 1]:cu_seqlens[i]] = 0
        return attention_mask
    
    def get_current_step(self):
        return self.step

    def forward_embed(self, tokens: np.ndarray, ):
        self.token_len = tokens.shape[1]
        self.tokens = tokens[0].tolist()

        input_ids = np.zeros((tokens.shape[0], self.MAX_INPUT_LENGTH), dtype=type_convert(self.input_tensors[self.name_embed][0].dtype()))
        input_ids[:, :min(self.MAX_INPUT_LENGTH, tokens.shape[1])] = tokens
        self.input_tensors[self.name_embed][0].update_data(input_ids)
        self.net.process(self.name_embed, self.input_tensors[self.name_embed], self.output_tensors[self.name_embed])
        for i in range(self.num_deepstack):
            self.deepstack_buffers[i].zeros()

    def forward_vit(self, pixel_values, position_ids, pos_idx, pos_weight, grid_thw, vit_offset):
        t, h, w = grid_thw.squeeze(0).tolist()
        real_patches = t * h * w
        assert pixel_values.size == real_patches * self.vit_hidden_states_input_shape[1], \
            f"pixel_values size {pixel_values.size} != {real_patches}*{self.vit_hidden_states_input_shape[1]}"
        assert position_ids.size == real_patches * 2, \
            f"position_ids size {position_ids.size} != {real_patches}*2"
        assert pos_idx.size == real_patches * 4, \
            f"pos_idx size {pos_idx.size} != {real_patches}*4"
        assert pos_weight.size == real_patches * 4, \
            f"pos_weight size {pos_weight.size} != {real_patches}*4"

        
        if self.is_vit_dynamic:
            pixel_values = pixel_values.astype(type_convert(self.input_tensors[self.name_vit][0].dtype()))
            position_ids = position_ids.astype(type_convert(self.input_tensors[self.name_vit][1].dtype()))
            pos_idx = pos_idx.astype(type_convert(self.input_tensors[self.name_vit][2].dtype()))
            pos_weight = pos_weight.astype(type_convert(self.input_tensors[self.name_vit][3].dtype()))
            pos_weight = np.expand_dims(pos_weight, axis=-1)
            
            pixel_values_full = pixel_values[:real_patches, :]
            pos_ids_full = position_ids[:real_patches, :]
            pos_idx_full = pos_idx[:real_patches, :]
            pos_weight_full = pos_weight[:real_patches, :, :]
            vit_attention_mask = np.zeros((1, 1, real_patches, real_patches), dtype=np.float32)

            self.input_tensors[self.name_vit][0].reshape(pixel_values_full.shape)
            self.input_tensors[self.name_vit][0].update_data(pixel_values_full)
            self.input_tensors[self.name_vit][1].reshape(pos_ids_full.shape)
            self.input_tensors[self.name_vit][1].update_data(pos_ids_full)
            self.input_tensors[self.name_vit][2].reshape(pos_idx_full.shape)
            self.input_tensors[self.name_vit][2].update_data(pos_idx_full)
            self.input_tensors[self.name_vit][3].reshape(pos_weight_full.shape)
            self.input_tensors[self.name_vit][3].update_data(pos_weight_full)
            self.input_tensors[self.name_vit][4].update_data(vit_attention_mask)
        else:
            pixel_values = pixel_values.astype(type_convert(self.input_tensors[self.name_vit][0].dtype()))
            pixel_values_prefill = np.zeros(self.vit_hidden_states_input_shape, dtype=type_convert(self.input_tensors[self.name_vit][0].dtype()))
            pixel_values_prefill[:pixel_values.shape[0],:] = pixel_values

            position_ids = position_ids.astype(type_convert(self.input_tensors[self.name_vit][1].dtype()))
            pos_ids_prefill = np.zeros(self.vit_pos_ids_input_shape, dtype=type_convert(self.input_tensors[self.name_vit][1].dtype()))
            pos_ids_prefill[:position_ids.shape[0],:] = position_ids

            pos_idx = pos_idx.astype(type_convert(self.input_tensors[self.name_vit][2].dtype()))
            pos_idx_prefill = np.zeros(self.pos_idx_input_shape, dtype=type_convert(self.input_tensors[self.name_vit][2].dtype()))
            pos_idx_prefill[:pos_idx.shape[0],:] = pos_idx

            pos_weight = pos_weight.astype(type_convert(self.input_tensors[self.name_vit][3].dtype()))
            pos_weight = np.expand_dims(pos_weight, axis=-1)
            pos_weight_prefill = np.zeros(self.pos_weight_input_shape, dtype=type_convert(self.input_tensors[self.name_vit][3].dtype()))
            pos_weight_prefill[:pos_weight.shape[0], :, :] = pos_weight
            
            vit_attention_mask = np.full((1, 1, self.MAX_PATCHES, self.MAX_PATCHES), -10000.0, dtype=np.float32)
            vit_attention_mask[:, :, :real_patches, :real_patches] = 0.0
            
            self.input_tensors[self.name_vit][0].update_data(pixel_values_prefill)
            self.input_tensors[self.name_vit][1].update_data(pos_ids_prefill)
            self.input_tensors[self.name_vit][2].update_data(pos_idx_prefill)
            self.input_tensors[self.name_vit][3].update_data(pos_weight_prefill)
            self.input_tensors[self.name_vit][4].update_data(vit_attention_mask)

        self.net.process(self.name_vit, self.input_tensors[self.name_vit], self.output_tensors[self.name_vit])
        
        dst_offset = vit_offset * self.hidden_size
        vit_size   = (real_patches // 4)   * self.hidden_size
        vision_embeds = self.output_tensors[self.name_vit][0].asnumpy().reshape(-1)
        token_embeds = self.output_tensors[self.name_embed][0].asnumpy().reshape(-1)
        token_embeds[dst_offset : dst_offset + vit_size] = vision_embeds[:vit_size]
        token_embeds = token_embeds.reshape(self.output_tensors[self.name_embed][0].asnumpy().shape)
        
        self.output_tensors[self.name_embed][0].update_data(token_embeds)

        target_shape_copy = self.output_tensors[self.name_embed][0].asnumpy().shape
        for i in range(self.num_deepstack):
            vision_embeds_copy = self.output_tensors[self.name_vit][i + 1].asnumpy().reshape(-1)
            deepstack_data = self.deepstack_buffers[i].asnumpy().reshape(-1)
            deepstack_data[dst_offset : dst_offset + vit_size] = vision_embeds_copy[:vit_size]
            deepstack_data = deepstack_data.reshape(target_shape_copy)
            self.deepstack_buffers[i].update_data(deepstack_data)
        self.vit_run = True

    def sample_token(self):
        # lmhead_with_topk( = lm_head + greedy_head )
        if self.generation_mode is None:
            token = int(self.output_tensors[self.name_lm][0].asnumpy().item())
        # lm_head + greedy_head
        elif self.generation_mode == "greedy":
            self.input_tensors[self.greedy][0] = self.output_tensors[self.name_lm][0]
            self.net.process(self.greedy, self.input_tensors[self.greedy], self.output_tensors[self.greedy])
            token = int(self.output_tensors[self.greedy][0].asnumpy())
        # lm_head + sample_head
        elif self.generation_mode == "sample":
            self.input_tensors[self.sample][0] = self.output_tensors[self.name_lm][0]
            generated_tokens = np.zeros([1, self.MAX_INPUT_LENGTH], type_convert(self.input_tensors[self.sample][1].dtype()))
            generated_tokens[0, :len(self.tokens)] = self.tokens
            self.input_tensors[self.sample][1].update_data(generated_tokens)
            self.input_tensors[self.sample][2].update_data([self.repeat_penalty])
            self.input_tensors[self.sample][3].update_data([self.temperature])
            top_k = np.ones([1], type_convert(self.input_tensors[self.sample][4].dtype())) * self.top_k
            self.input_tensors[self.sample][4].update_data(top_k)
            self.input_tensors[self.sample][5].update_data([self.top_p])
            self.net.process(self.sample, self.input_tensors[self.sample], self.output_tensors[self.sample])

            probs = self.output_tensors[self.sample][0].asnumpy()[0, :self.top_k]
            token_TopK = self.output_tensors[self.sample][1].asnumpy()[0, :self.top_k]
            token = int(np.random.choice(token_TopK, p=probs / probs.sum()))
        else:
            raise ValueError("Invalid generation_mode parameter. Supported options are 'greedy' and 'sample'.")
        
        self.tokens.append(token)
        return token

    def forward_first(self, position_ids):
        self.token_pos_length = position_ids.max() + 1
        position_ids = position_ids.flatten()

        ATTENTION_MASK = 0xC61C
        if self.is_dynamic:
            attention_mask = [ATTENTION_MASK] * (self.token_len * self.token_len)
            for i in range(self.token_len):
                for j in range(i + 1):
                    attention_mask[i * self.token_len + j] = 0
            attention_mask = np.array(attention_mask, dtype=type_convert(self.input_tensors[self.name_blocks[0]][2].dtype())).reshape(1, 1, self.token_len, self.token_len)
            position_ids_pad = np.array(position_ids, dtype=type_convert(self.input_tensors[self.name_blocks[0]][1].dtype())).reshape(3, self.token_len)

            self.input_tensors[self.name_blocks[0]][0].reshape([1, self.token_len, self.hidden_size])
            self.input_tensors[self.name_blocks[0]][1].reshape([3, self.token_len])
            self.input_tensors[self.name_blocks[0]][2].reshape([1, 1, self.token_len, self.token_len])
            self.input_tensors[self.name_blocks[0]][0] = sail.Tensor(self.output_tensors[self.name_embed][0], [1, self.token_len, self.hidden_size], 0)
        else:
            attention_mask = [ATTENTION_MASK] * (self.MAX_INPUT_LENGTH * self.MAX_INPUT_LENGTH)
            for i in range(self.token_len):
                for j in range(self.token_len):
                    if j <= i:
                        attention_mask[i * self.MAX_INPUT_LENGTH + j] = 0
            attention_mask = np.array(attention_mask, dtype=type_convert(self.input_tensors[self.name_blocks[0]][2].dtype())).reshape(self.input_tensors[self.name_blocks[0]][2].shape())

            position_ids_pad = [0] * (3 * self.MAX_INPUT_LENGTH)
            ori_length = len(position_ids) // 3
            for i in range(3):
                ori_offset = i * ori_length
                dst_offset = i * self.MAX_INPUT_LENGTH
                position_ids_pad[dst_offset : dst_offset + ori_length] = \
                    position_ids[ori_offset : ori_offset + ori_length]
            position_ids_pad = np.array(position_ids_pad, dtype=type_convert(self.input_tensors[self.name_blocks[0]][1].dtype())).reshape(self.input_tensors[self.name_blocks[0]][1].shape())
            
            self.input_tensors[self.name_blocks[0]][0] = self.output_tensors[self.name_embed][0]


        self.input_tensors[self.name_blocks[0]][1].update_data(position_ids_pad)
        self.input_tensors[self.name_blocks[0]][2].update_data(attention_mask)
        for i in range(self.num_layers):
            block_output_tensors = { \
                0: self.first_hidden_states_output, \
                1: self.past_key_input[i], \
                2: self.past_value_input[i], \
            }
            self.net.process(self.name_blocks[i], self.input_tensors[self.name_blocks[0]], block_output_tensors)
            if self.is_dynamic:
                self.input_tensors[self.name_blocks[0]][0] = sail.Tensor(self.first_hidden_states_output, [1, self.token_len, self.hidden_size], 0)
            else:
                self.input_tensors[self.name_blocks[0]][0].sync_d2d( \
                    self.first_hidden_states_output, \
                    0, \
                    0, \
                    len(self.first_hidden_states_output), \
                )
            if (self.vit_run and (i < self.num_deepstack)):
                copy_size = self.MAX_INPUT_LENGTH * self.hidden_size
                self.input_tensors[self.name_add][0].sync_d2d(self.first_hidden_states_output ,0 , 0, copy_size)
                self.input_tensors[self.name_add][1].sync_d2d(self.deepstack_buffers[i] ,0 , 0, copy_size)
                self.net.process(self.name_add, self.input_tensors[self.name_add], self.output_tensors[self.name_add])
                self.first_hidden_states_output.sync_d2d(self.output_tensors[self.name_add][0] ,0 , 0, copy_size)
        
        self.vit_run = False
        
        # linear process
        self.step = self.token_len
        offset_bytes = self.first_hidden_states_output.size() // self.MAX_INPUT_LENGTH
        self.token_pos_length = position_ids.max() + 1
        self.input_tensors[self.name_lm][0].sync_d2d(self.first_hidden_states_output, \
                    (self.token_len - 1) * offset_bytes, \
                    0, \
                    offset_bytes , \
        )

        self.net.process(self.name_lm, self.input_tensors[self.name_lm], self.output_tensors[self.name_lm])

        # sample
        self.last_id = self.sample_token()
        self.logger.debug(f"{Logger.file_lineno()} get first inference results token id {self.last_id}")
        return self.last_id

    # The following tokens prediction
    def forward_next(self, position_id):
        # embedding prepare inputs & infer
        self.input_tensors[self.name_embed_cache][0].update_data(np.array([self.last_id], \
                        dtype=type_convert(self.input_tensors[self.name_embed_cache][0].dtype())).reshape( \
                            self.input_tensors[self.name_embed_cache][0].shape()))
        self.net.process(self.name_embed_cache, self.input_tensors[self.name_embed_cache], \
                        self.output_tensors[self.name_embed_cache])
        
        # transformer block prepare inputs & process
        causal_mask = np.zeros(self.next_attention_mask_input.shape(), dtype=np.float32)
        for batch_idx in range(self.next_attention_mask_input.shape()[0]):
            causal_mask[batch_idx, 0, :, self.step:-1] = float("-10000")
        position_ids = np.array([position_id], dtype=np.int32).reshape(3, 1)
        self.next_hidden_states_input.sync_d2d( \
            self.output_tensors[self.name_embed_cache][0],
            0, \
            0, \
            self.output_tensors[self.name_embed_cache][0].shape()[-1], \
        )
        self.next_pos_ids_input.update_data(position_ids)
        if type_convert(self.next_attention_mask_input.dtype()) == np.uint16:
            causal_mask = torch.from_numpy(causal_mask).type(torch.bfloat16).view(torch.uint16).numpy()
        self.next_attention_mask_input.update_data(causal_mask)
        block_output_tensors = { \
            0: self.next_hidden_states_output, \
            1: self.present_key_output, \
            2: self.present_value_output, \
        }
        for i in range(self.num_layers):
            block_input_tensors = { \
                0: self.next_hidden_states_input, \
                1: self.next_pos_ids_input, \
                2: self.next_attention_mask_input, \
                3: self.past_key_input[i], \
                4: self.past_value_input[i], \
            }
            self.net.process(self.name_blocks_cache[i], block_input_tensors, block_output_tensors)
            self.next_hidden_states_input.sync_d2d( \
                self.next_hidden_states_output, \
                0, \
                0, \
                len(self.next_hidden_states_output), \
            )
            for batch_idx in range(self.past_key_input[i].shape()[0]):
                self.past_key_input[i].sync_d2d(self.present_key_output, \
                            batch_idx * self.past_kv_stride[0],
                            batch_idx * self.past_kv_stride[0] + self.step * self.past_kv_stride[1],
                            self.past_kv_stride[1])
                self.past_value_input[i].sync_d2d(self.present_value_output, \
                            batch_idx * self.past_kv_stride[0],
                            batch_idx * self.past_kv_stride[0] + self.step * self.past_kv_stride[1],
                            self.past_kv_stride[1])

        # linear process
        self.step += 1
        self.token_pos_length += 1
        self.input_tensors[self.name_lm][0].sync_d2d(self.next_hidden_states_output, \
                    0, \
                    0, \
                    len(self.next_hidden_states_output), \
        )
        self.net.process(self.name_lm, self.input_tensors[self.name_lm], self.output_tensors[self.name_lm])

        # sample
        self.last_id = self.sample_token()
        self.logger.debug(f"{Logger.file_lineno()} get step {self.step} inference results token id {self.last_id}")

        return self.last_id

    def forward_prefill(self, position_ids):
        if self.step == 0 or not self.support_history:
            self.history_max_posid = 0
            return self.forward_first(position_ids)
        self.max_posid += self.history_max_posid
        position_ids = position_ids + self.history_max_posid
        return self.forward_first(position_ids)
    

    def process(self, messages, media_type):
        if media_type == "text":
            return self.processor.apply_chat_template(messages,
                                                      tokenize=True,
                                                      add_generation_prompt=True,
                                                      return_dict=True,
                                                      return_tensors="pt")
        text = self.processor.apply_chat_template(messages,
                                                  tokenize=False,
                                                  add_generation_prompt=True)
        images, videos, video_kwargs = process_vision_info(messages,
                                                           image_patch_size=16,
                                                           return_video_kwargs=True,
                                                           return_video_metadata=True)
        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
        else:
            video_metadatas = None
        return self.processor(text=[text],
                              images=images,
                              videos=videos,
                              video_metadata=video_metadatas,
                              do_resize=False,
                              return_tensors="pt",
                              **video_kwargs)
    
    def chat(self):
        """
        Start a chat session.
        """
        # Instruct
        print("""\n=================================================================
1. If you want to quit, please enter one of [q, quit, exit]
2. To create a new chat session, please enter one of [clear, new]
3. To change the image/video, please enter one of [change, changeinput]
=================================================================""")
        # Stop Chatting with "exit" input
        while True:
            self.input_str = input("\nQuestion: ")
            # Quit
            if self.input_str in ["exit", "q", "quit"]:
                break
            if self.input_str in ["clear", "new", "c"]:
                print("New chat session created.")
                self.clear_history()
                self.history_max_posid = 0
                self.media_path = None
                continue
            # Change media path
            if self.input_str in ["change", "changeinput"]:
                new_path = input("\nEnter new Image or Video Path (or leave empty for text-only): ")
                new_path = new_path.strip()
                if new_path == "":
                    self.media_path = "" # 设为空字符串，用于纯文本模式
                    print("Media path cleared. Switched to text-only mode.")
                elif not os.path.exists(new_path):
                    print(f"Can't find {new_path}. Media path not changed.")
                else:
                    self.media_path = new_path # 更新缓存的路径
                    print(f"Media path changed to: {self.media_path}")
                continue # 重新循环，等待输入 Question
            if self.media_path is None:
                # 第一次运行或 "clear" 之后，提示输入路径
                media_path = input("\nImage or Video Path: ")
                media_path = media_path.strip()
                self.media_path = media_path # 缓存路径供下次使用
            else:
                # 已经有缓存路径，直接使用
                media_path = self.media_path
                if media_path != "":
                    print(f"Using cached media: {media_path}")
                else:
                    print("Using text-only mode.")
            
            if media_path == "":
                messages = self.text_message()
                media_type = "text"
            elif not os.path.exists(media_path):
                print("Can't find image or video: {}".format(media_path))
                if media_path == self.media_path:
                    self.media_path = None
                    print("Cached media path was invalid and has been cleared.")
                continue
            else:
                media_type = self.get_media_type(media_path)
                if media_type == "image":
                    messages = self.image_message(media_path)
                elif media_type == "video":
                    messages = self.video_message(media_path)
                else:
                    print("Unsupported media type: {}".format(media_path))
                    continue

            inputs = self.process(messages, media_type)
            token_len = inputs.input_ids.numel()
            if token_len > self.MAX_INPUT_LENGTH:
                if media_type in ["image", "video"]:
                    print("grid_thw:{}".format(inputs.image_grid_thw if media_type ==
                                               "image" else inputs.video_grid_thw))
                print(
                    "Error: The maximum question length should be shorter than {} but we get {} instead."
                    .format(self.MAX_INPUT_LENGTH, token_len))
                continue
            if self.support_history:
                if (token_len + self.step > self.seq_len - 128) or \
                (self.step > self.PREFILL_KV_LENGTH):
                    print("Warning: History is full and clear it to continue.")
                    self.clear_history()
                    self.history_max_posid = 0
            print("\nAnswer:")

            # Chat
            first_start = time.time()
            self.forward_embed(inputs.input_ids.numpy())
            if media_type == "image":
                vit_start = time.time()
                self.vit_process_image(inputs)
                vit_end = time.time()
                position_ids = self.get_rope_index(inputs.input_ids, inputs.image_grid_thw,
                                                   self.ID_IMAGE_PAD)
                self.max_posid = int(position_ids.max())
                token = self.forward_prefill(position_ids.numpy())
            elif media_type == "video":
                vit_start = time.time()
                self.vit_process_video(inputs)
                vit_end = time.time()
                position_ids = self.get_rope_index(inputs.input_ids, inputs.video_grid_thw,
                                                   self.ID_VIDEO_PAD)
                self.max_posid = int(position_ids.max())
                token = self.forward_prefill(position_ids.numpy())
            else:
                position_ids = 3 * [i for i in range(token_len)]
                self.max_posid = token_len - 1
                token = self.forward_prefill(np.array(position_ids, dtype=np.int32))
            first_end = time.time()
            tok_num = 0
            # Following tokens
            full_word_tokens = []
            text = ""
            while token not in [self.ID_IM_END, self.ID_END
                                ] and self.step < self.seq_len:
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
                self.max_posid += 1
                position_ids = np.array([self.max_posid, self.max_posid, self.max_posid],
                                        dtype=np.int32)
                token = self.forward_next(position_ids)
                tok_num += 1
            self.history_max_posid = self.max_posid + 2
            next_end = time.time()
            first_duration = first_end - first_start
            next_duration = next_end - first_end
            tps = tok_num / next_duration
            print(f"\nFTL: {first_duration:.3f} s")
            print(f"TPS: {tps:.3f} token/s")
            if media_type == "image":
                print(f"Vision({inputs.image_grid_thw.tolist()}): {vit_end - vit_start:.3f} s")
            elif media_type == "video":
                print(f"Vision({inputs.video_grid_thw.tolist()}): {vit_end - vit_start:.3f} s")


def main(args):
    model = Qwen3_VL(args)
    model.chat()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # yapf: disable
    parser.add_argument('-m', '--model_path', type=str, required=True,
                        help='path to the bmodel file')
    parser.add_argument('-c', '--config_path', type=str, default="../config",
                        help='path to the processor file')
    parser.add_argument('-vr', '--video_ratio', type=float, default=0.25, help='Set video ratio, default is 0.25')
    parser.add_argument('-d', '--devid', type=int, default=0, help='device ID to use')
    parser.add_argument('-ll',
                        '--log_level',
                        type=str,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        default="INFO",
                        help='log level, default: INFO, option[DEBUG, INFO, WARNING, ERROR]')
    parser.add_argument('--do_sample', 
                        action='store_true', 
                        help="if set, generate tokens by sample parameters")
    # yapf: enable
    args = parser.parse_args()
    main(args)
