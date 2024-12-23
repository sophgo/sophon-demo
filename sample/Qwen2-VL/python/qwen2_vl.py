import time
import argparse
from sophon import sail
from transformers import AutoProcessor, AutoTokenizer, Qwen2VLConfig, BatchFeature
from transformers.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessorKwargs
from vision_process import process_vision_info
import json
import os
import torch
from typing import Optional, Tuple
from typing import List
import numpy as np
import torch.nn.functional as F

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

class Qwen2VLInputProcessor:
    """ overwrite <class 'transformers.models.qwen2_vl.processing_qwen2_vl.Qwen2VLProcessor'>.__call__ function
            to use exsiting image_grid_thw or video_grid_thw
    """
    def __init__(self, processor_path, trust_remote_code=True, **kwargs):
        self.processor = AutoProcessor.from_pretrained(processor_path,
                                                       trust_remote_code=trust_remote_code, **kwargs)
    
    def __call__(
        self,
        images = None,
        text = None,
        videos = None,
        image_grid_thw = None,
        video_grid_thw = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to Qwen2TokenizerFast's [`~Qwen2TokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the vision inputs, this method forwards the `vision_infos` and `kwrags` arguments to
        Qwen2VLImageProcessor's [`~Qwen2VLImageProcessor.__call__`] if `vision_infos` is not `None`.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported. If given, `image_grid_thw` will be
                overwirted.
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            videos (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of videos to be prepared. Each video can be a 4D NumPy array or PyTorch
                tensor, or a nested list of 3D frames. Both channels-first and channels-last formats are supported. If given, 
                `video_grid_thw` will be overwirted.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **pixel_values_videos** -- Pixel values of videos to be fed to a model. Returned when `videos` is not `None`.
            - **image_grid_thw** -- List of image 3D grid in LLM. Returned when `images` is not `None`.
            - **video_grid_thw** -- List of video 3D grid in LLM. Returned when `videos` is not `None`.
        """
        output_kwargs = self.processor._merge_kwargs(
            Qwen2VLProcessorKwargs,
            tokenizer_init_kwargs=self.processor.tokenizer.init_kwargs,
            **kwargs,
        )
        if images is not None:
            image_inputs = self.processor.image_processor(images=images, videos=None, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = {}

        if videos is not None:
            video_inputs = self.processor.image_processor(images=None, videos=videos, **output_kwargs["videos_kwargs"])
            video_grid_thw = video_inputs["video_grid_thw"]
        else:
            video_inputs = {}

        if not isinstance(text, list):
            text = [text]

        if image_grid_thw is not None:
            merge_length = self.processor.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while "<|image_pad|>" in text[i]:
                    text[i] = text[i].replace(
                        "<|image_pad|>", "<|placeholder|>" * (image_grid_thw[index].prod() // merge_length), 1
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", "<|image_pad|>")

        if video_grid_thw is not None:
            merge_length = self.processor.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while "<|video_pad|>" in text[i]:
                    text[i] = text[i].replace(
                        "<|video_pad|>", "<|placeholder|>" * (video_grid_thw[index].prod() // merge_length), 1
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", "<|video_pad|>")

        _ = output_kwargs["text_kwargs"].pop("padding_side", None)
        text_inputs = self.processor.tokenizer(text, **output_kwargs["text_kwargs"])

        return BatchFeature(data={**text_inputs, **image_inputs, **video_inputs})

    def apply_chat_template(self, msg, tokenize=False, add_generation_prompt=True, **kwargs):
        return self.processor.apply_chat_template(msg, tokenize=tokenize, add_generation_prompt=add_generation_prompt, **kwargs)


class Qwen2VL():

    def __init__(self, **kwargs):
        # devid
        self.dev_id = kwargs.get("dev_id", 0)
        self.handle = sail.Handle(self.dev_id)
        self.net = sail.EngineLLM(kwargs["bmodel_path"], [self.dev_id])

        # graph
        self.graph_names = self.net.get_graph_names()

        # initialize qwen parameters
        self.num_layers = 0
        for graph_name in self.graph_names:
            if "block_cache_" in graph_name:
                self.num_layers += 1
        self.first_hidden_states_input_shape = self.net.get_input_shape("block_0", 0)
        self.vit_hidden_states_input_shape = self.net.get_input_shape("vit", 0)
        self.vit_pos_ids_input_shape = self.net.get_input_shape("vit", 1)
        self.vit_attention_mask_input_shape = self.net.get_input_shape("vit", 2)
        _, self.seq_len, self.hidden_size = self.first_hidden_states_input_shape
        self.vision_seq_len = self.vit_hidden_states_input_shape[0]
        self.input_tensors = {}
        self.output_tensors = {}
        self.past_kv_stride = [1] * len(self.net.get_input_shape("block_cache_0", 3))
        for dim_i in range(len(self.net.get_input_shape("block_cache_0", 3))-2, -1, -1):
            self.past_kv_stride[dim_i] = self.net.get_input_shape("block_cache_0", 3)[dim_i + 1] * \
                                            self.past_kv_stride[dim_i + 1]

        # initialize net name
        self.is_greedy_sample = True
        self.name_embed = "embedding"
        self.name_embed_cache = "embedding_cache"
        self.name_lm = "lm_head"
        self.name_blocks = ["block_"+str(i) for i in range(self.num_layers)]
        self.name_blocks_cache = ["block_cache_"+str(i) for i in range(self.num_layers)]
        self.name_sample = "greedy_head" if self.is_greedy_sample else ""
        self.name_penalty = "penalty_sample_head" if self.is_greedy_sample else ""
        self.name_vit = "vit"

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

        # sample tensors (inputs & outputs)
        self.output_tensors[self.name_sample] = self.net.create_max_output_tensors(self.name_sample)

        # init preprocessor & tokenizer & configs
        self.processor = Qwen2VLInputProcessor(args.processor_path,
                                                       trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path,
                                                       trust_remote_code=True)
        with open(args.config, 'r') as f:
            self.config = json.load(f)
        self.loaded_config = Qwen2VLConfig(**self.config)
        self.ID_END = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        self.ID_IM_END = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

        # init runtime val
        self.init_runtime_vals()

    def init_runtime_vals(self):
        self.step = 0
        self.token_pos_length = 0
        self.last_id = None

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

    def get_rope_index(
            self,
            input_ids: torch.LongTensor,
            image_grid_thw: Optional[torch.LongTensor] = None,
            video_grid_thw: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            spatial_merge_size = self.loaded_config.vision_config.spatial_merge_size
            image_token_id = self.loaded_config.image_token_id
            video_token_id = self.loaded_config.video_token_id
            vision_start_token_id = self.loaded_config.vision_start_token_id
            mrope_position_deltas = []
            if image_grid_thw is not None or video_grid_thw is not None:
                total_input_ids = input_ids
                position_ids = torch.ones(
                    3, input_ids.shape[0], input_ids.shape[1], dtype=torch.long, device=input_ids.device
                )
                image_index, video_index = 0, 0
                for i, input_ids in enumerate(total_input_ids):
                    if attention_mask is not None:
                        input_ids = input_ids[attention_mask[i] == 1]
                    image_nums, video_nums = 0, 0
                    vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                    vision_tokens = input_ids[vision_start_indices + 1]
                    image_nums = (vision_tokens == image_token_id).sum()
                    video_nums = (vision_tokens == video_token_id).sum()
                    input_tokens = input_ids.tolist()
                    llm_pos_ids_list: list = []
                    st = 0
                    remain_images, remain_videos = image_nums, video_nums
                    for _ in range(image_nums + video_nums):
                        if image_token_id in input_tokens and remain_images > 0:
                            ed_image = input_tokens.index(image_token_id, st)
                        else:
                            ed_image = len(input_tokens) + 1
                        if video_token_id in input_tokens and remain_videos > 0:
                            ed_video = input_tokens.index(video_token_id, st)
                        else:
                            ed_video = len(input_tokens) + 1
                        if ed_image < ed_video:
                            t, h, w = (
                                image_grid_thw[image_index][0],
                                image_grid_thw[image_index][1],
                                image_grid_thw[image_index][2],
                            )
                            image_index += 1
                            remain_images -= 1
                            ed = ed_image
                        else:
                            t, h, w = (
                                video_grid_thw[video_index][0],
                                video_grid_thw[video_index][1],
                                video_grid_thw[video_index][2],
                            )
                            video_index += 1
                            remain_videos -= 1
                            ed = ed_video
                        llm_grid_t, llm_grid_h, llm_grid_w = (
                            t.item(),
                            h.item() // spatial_merge_size,
                            w.item() // spatial_merge_size,
                        )
                        text_len = ed - st

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                        t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                        h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                        w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                        llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                        st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                    if st < len(input_tokens):
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        text_len = len(input_tokens) - st
                        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                    position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                    mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
                mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
                return position_ids, mrope_position_deltas
            else:
                if attention_mask is not None:
                    position_ids = attention_mask.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attention_mask == 0, 1)
                    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
                    max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                    mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
                else:
                    position_ids = (
                        torch.arange(input_ids.shape[1], device=input_ids.device)
                        .view(1, 1, -1)
                        .expand(3, input_ids.shape[0], -1)
                    )
                    mrope_position_deltas = torch.zeros(
                        [input_ids.shape[0], 1],
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    )

                return position_ids, mrope_position_deltas

    def preprocess(
        self, 
        messages, 
        image_grid_thw=None, 
        video_grid_thw=None,
    ):
        # check, only support a input type
        if image_grid_thw is not None:
            assert video_grid_thw is None
        if video_grid_thw is not None:
            assert image_grid_thw is None

        # init prompt
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        
        if image_grid_thw is None and video_grid_thw is None:
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        else:
            inputs = self.processor(
                text=texts,
                images=None,
                videos=None,
                image_grid_thw = image_grid_thw,
                video_grid_thw = video_grid_thw,
                padding=True,
                return_tensors="pt",
            )

        if self.seq_len < inputs.input_ids.shape[-1]:
            raise ValueError(
                    f"The input_length must be shorter than model's seq_length (got `input_length`: {inputs.input_ids.shape[-1]}"
                    f" and `seq_length`: {self.seq_len})."
                )

        input_ids = inputs.input_ids
        if image_grid_thw is None and video_grid_thw is None:
            if "image_grid_thw" in inputs:
                image_grid_thw = inputs.image_grid_thw
                video_grid_thw = None
            else:
                image_grid_thw = None
                video_grid_thw = inputs.video_grid_thw
        input_ids_prefill = torch.zeros(input_ids.shape[0], self.seq_len).to(torch.int32)
        input_ids_prefill[:, :input_ids.shape[-1]] = input_ids
        attention_mask_prefill = torch.zeros(inputs.attention_mask.shape[0], self.seq_len)
        attention_mask_prefill[:, :input_ids.shape[-1]] = inputs.attention_mask
        image_mask = (input_ids_prefill == self.loaded_config.image_token_id)
        true_indices = torch.nonzero(image_mask, as_tuple=True)[1]

        if true_indices.numel() > 0:
            first_true_index = true_indices[0].item()
        else:
            first_true_index = None
        
        position_ids, _ = self.get_rope_index(
            input_ids_prefill, image_grid_thw, video_grid_thw, attention_mask_prefill
        )

        return position_ids, inputs, first_true_index

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = np.tile(np.array(list(range(h)))[:, None], (1, w))
            hpos_ids = hpos_ids.reshape(
                h // self.loaded_config.vision_config.spatial_merge_size,
                self.loaded_config.vision_config.spatial_merge_size,
                w // self.loaded_config.vision_config.spatial_merge_size,
                self.loaded_config.vision_config.spatial_merge_size,
            )
            hpos_ids = hpos_ids.transpose(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = np.tile(np.array(list(range(w)))[None], (h, 1))
            wpos_ids = wpos_ids.reshape(
                h // self.loaded_config.vision_config.spatial_merge_size,
                self.loaded_config.vision_config.spatial_merge_size,
                w // self.loaded_config.vision_config.spatial_merge_size,
                self.loaded_config.vision_config.spatial_merge_size,
            )
            wpos_ids = wpos_ids.transpose(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(np.tile(np.stack([hpos_ids, wpos_ids], axis=-1), (t, 1)))
        pos_ids = np.concatenate(pos_ids, axis=0)
        return pos_ids

    def get_vision_mask(self, grid_thw):
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        seq_length = cu_seqlens.max()

        attention_mask = torch.zeros([1, seq_length, seq_length], device="cpu", dtype=torch.bool)
        for i in range(1, cu_seqlens.shape[0]):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True
        return attention_mask

    def vision_process(
            self, 
            pixel_values_images,
            pixel_values_videos: np.ndarray, 
            image_grid_thw,
            video_grid_thw):
        if pixel_values_images is not None:
            pixel_values = pixel_values_images
            grid_thw = image_grid_thw
            input_type = "image"
        else:
            pixel_values = pixel_values_videos
            grid_thw = video_grid_thw
            input_type = "video"
        # ViT prepare inputs & infer
        real_len = pixel_values.shape[0]
        pixel_values_prefill = np.zeros(self.vit_hidden_states_input_shape, dtype=type_convert(self.input_tensors[self.name_vit][0].dtype()))
        pixel_values_prefill[:pixel_values.shape[0],:] = pixel_values
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        pos_ids_prefill = np.zeros(self.vit_pos_ids_input_shape, dtype=type_convert(self.input_tensors[self.name_vit][1].dtype()))
        pos_ids_prefill[:rotary_pos_emb.shape[0],:] = rotary_pos_emb
        pos_ids_prefill = pos_ids_prefill
        visual_attention_mask = self.get_vision_mask(torch.from_numpy(grid_thw))
        visual_attention_mask_fp = torch.zeros((visual_attention_mask.shape), dtype=torch.float32)
        visual_attention_mask = visual_attention_mask_fp.masked_fill(visual_attention_mask.logical_not(), -10000).numpy()
        visual_attention_mask_prefill = np.zeros(self.vit_attention_mask_input_shape, dtype=type_convert(self.input_tensors[self.name_vit][2].dtype()))
        visual_attention_mask_prefill[:, :visual_attention_mask.shape[1], :visual_attention_mask.shape[2]] = visual_attention_mask
        self.input_tensors[self.name_vit][0].update_data(pixel_values_prefill)
        self.input_tensors[self.name_vit][1].update_data(pos_ids_prefill)
        self.input_tensors[self.name_vit][2].update_data(visual_attention_mask_prefill)
        self.net.process(self.name_vit, self.input_tensors[self.name_vit], self.output_tensors[self.name_vit])
        vision_embeds = torch.from_numpy(self.output_tensors[self.name_vit][0].asnumpy()[:real_len//4]).type( \
                                            torch.bfloat16).view(torch.uint16).detach().numpy() # uint16(bfloat16)
        if input_type == "image":
            return vision_embeds, None
        else:
            return None, vision_embeds

    # inference for the first token
    def forward_first(
            self, 
            tokens: np.ndarray, 
            position_ids: np.ndarray,
            image_embeds: np.ndarray,
            video_embeds: np.ndarray,):
        # self.init_kv_cache(tokens.shape[0])
        input_ids = np.zeros((tokens.shape[0], self.seq_len), dtype=type_convert(self.input_tensors[self.name_embed][0].dtype()))
        input_ids[:, :min(self.seq_len, tokens.shape[1])] = tokens
        token_length = tokens.shape[1]
        
        # embedding prepare inputs & infer
        # inputs_embeds = self.embed_tokens(torch.from_numpy(input_ids)).numpy()
        self.input_tensors[self.name_embed][0].update_data(input_ids)
        self.net.process(self.name_embed, self.input_tensors[self.name_embed], self.output_tensors[self.name_embed])
        inputs_embeds = self.output_tensors[self.name_embed][0].asnumpy() # uint16(bfloat16)
        
        # llm preprare inputs and outputs & infer
        if video_embeds is not None:
            video_mask = input_ids == self.loaded_config.video_token_id
            inputs_embeds[video_mask] = video_embeds
        else:
            image_mask = input_ids == self.loaded_config.image_token_id
            inputs_embeds[image_mask] = image_embeds
        padded_position_ids = np.zeros((3, position_ids.shape[1], self.seq_len), dtype=type_convert(self.input_tensors[self.name_blocks[0]][1].dtype())) 
        padded_position_ids[:, :, :position_ids.shape[-1]] = position_ids
        causal_mask = torch.zeros(self.input_tensors[self.name_blocks[0]][2].shape(), dtype=torch.float)
        temp_mask = torch.ones(self.seq_len, self.seq_len, dtype=torch.bool).tril(diagonal=0)
        for batch_idx in range(self.input_tensors[self.name_blocks[0]][2].shape()[0]):
            causal_mask[batch_idx, 0].masked_fill_(temp_mask.logical_not(), float("-10000"))
            causal_mask[batch_idx, 0, token_length:] = float("-10000")
            causal_mask[batch_idx, 0, :, token_length:] = float("-10000")
        if type_convert(self.input_tensors[self.name_blocks[0]][2].dtype()) == np.uint16:
            causal_mask = causal_mask.type(torch.bfloat16).view(torch.uint16).numpy()
        else:
            causal_mask = causal_mask.numpy().astype(type_convert(self.input_tensors[self.name_blocks[0]][2].dtype()))
        self.input_tensors[self.name_blocks[0]][0].update_data(inputs_embeds)
        self.input_tensors[self.name_blocks[0]][1].update_data(padded_position_ids)
        self.input_tensors[self.name_blocks[0]][2].update_data(causal_mask)
        
        # transformer block process
        for i in range(self.num_layers):
            block_output_tensors = { \
                0: self.first_hidden_states_output, \
                1: self.past_key_input[i], \
                2: self.past_value_input[i], \
            }
            self.net.process(self.name_blocks[i], self.input_tensors[self.name_blocks[0]], block_output_tensors)
            self.input_tensors[self.name_blocks[0]][0].sync_d2d( \
                self.first_hidden_states_output, \
                0, \
                0, \
                len(self.first_hidden_states_output), \
            )

        # linear process
        self.step = token_length
        self.token_pos_length = position_ids.max() + 1
        self.input_tensors[self.name_lm][0].sync_d2d(self.first_hidden_states_output, \
                    (token_length-1) * self.first_hidden_states_output.shape()[-1], \
                    0, \
                    self.first_hidden_states_output.shape()[-1], \
        )
        self.net.process(self.name_lm, self.input_tensors[self.name_lm], self.output_tensors[self.name_lm])

        # sample
        self.net.process(self.name_sample, self.output_tensors[self.name_lm], self.output_tensors[self.name_sample])
        self.last_id = self.output_tensors[self.name_sample][0].asnumpy().item()
        return self.last_id

    # The following tokens prediction
    def forward_next(self):
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
        position_ids = np.array([self.token_pos_length]*3, dtype=np.int32).reshape(3, 1, 1)
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
        self.net.process(self.name_sample, self.output_tensors[self.name_lm], self.output_tensors[self.name_sample])
        self.last_id = self.output_tensors[self.name_sample][0].asnumpy().item()
        return self.last_id

    def generate_message(self, input_type, paths, text, video_sample_num, **vision_configs):
        assert input_type in ["image", "video"]
        assert len(paths) > 0
        if input_type == "image":
            messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": input_type,
                                    input_type: path,
                                    **vision_configs,
                                } \
                                for path in paths
                            ],
                        }
            ]
        elif len(paths) == 1:
            messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": input_type,
                                    input_type: paths[0],
                                    "video_sample_num": video_sample_num,
                                    **vision_configs,
                                } \
                            ],
                        }
            ]
        else:
            messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": input_type,
                                    input_type: paths,
                                    **vision_configs,
                                }
                            ],
                        }
            ]
        messages[0]["content"].append({"type": "text", "text": text})
        return messages

    def is_end(self, token):
        return token in [self.ID_IM_END, self.ID_END
                                ] or self.step > self.seq_len


def main(args):
    model = Qwen2VL(dev_id=args.dev_id, bmodel_path=args.bmodel_path)
    video_embeds = None
    image_embeds = None
    video_grid_thw = None
    image_grid_thw = None
    pixel_values_images = None
    pixel_values_videos = None
    input_paths = args.input_paths
    input_type = args.input_type
    video_sample_num = args.frame_sample_num
    vision_preprocess_config = args.vision_preprocess_config
    if len(vision_preprocess_config) > 0:
        vision_preprocess_config = json.loads(vision_preprocess_config)
    else:
        vision_preprocess_config = {}

    print(
        "\n================================================================="
            "\n1. If you want to quit, please enter one of [q]"
            "\n=================================================================")
    
    while True:
        text = input("\nQuestion: ")
        if text == "q":
            break
        first_start = time.time()
        messages = model.generate_message(input_type, input_paths, text, video_sample_num, **vision_preprocess_config)
        messages = [messages]

        # preprocess text and images/video, get model inputs
        position_ids, inputs, image_offset = model.preprocess(messages=messages, \
                        video_grid_thw=video_grid_thw, image_grid_thw=image_grid_thw)
        if image_grid_thw is None and video_grid_thw is None:
            image_grid_thw = inputs.image_grid_thw if "image_grid_thw" in inputs else None
            video_grid_thw = inputs.video_grid_thw if "video_grid_thw" in inputs else None
            pixel_values_images = inputs.pixel_values if "pixel_values" in inputs else None
            pixel_values_videos = inputs.pixel_values_videos if "pixel_values_videos" in inputs else None

        # vision
        if image_embeds is None and video_embeds is None:
            image_embeds, video_embeds = model.vision_process(
                pixel_values_images=pixel_values_images.numpy() if pixel_values_images is not None else None,
                pixel_values_videos=pixel_values_videos.numpy() if pixel_values_videos is not None else None, 
                image_grid_thw=image_grid_thw.numpy() if image_grid_thw is not None else None,
                video_grid_thw=video_grid_thw.numpy() if video_grid_thw is not None else None)

        # Chat
        print("\nAnswer: ", end = '')
        token = model.forward_first(inputs.input_ids.numpy(), position_ids.numpy(), image_embeds, video_embeds)
        first_end = time.time()
        tok_num = 0
        # Following tokens
        full_word_tokens = []
        text = ""
        while not model.is_end(token):
            full_word_tokens.append(token)
            word = model.tokenizer.decode(full_word_tokens,
                                        skip_special_tokens=True)
            if "�" not in word:
                if len(full_word_tokens) == 1:
                    pre_word = word
                    word = model.tokenizer.decode(
                        [token, token],
                        skip_special_tokens=True)[len(pre_word):]
                text += word
                print(word, flush=True, end="")
                full_word_tokens = []
            token = model.forward_next()
            tok_num += 1
        next_end = time.time()
        first_duration = first_end - first_start
        next_duration = next_end - first_end
        tps = tok_num / next_duration
        print(f"\nFTL: {first_duration:.3f} s")
        print(f"TPS: {tps:.3f} token/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--bmodel_path',
                        type=str,
                        default="../models/BM1684X/qwen2-vl-7b_int4_seq512_1dev.bmodel",
                        help='path to the bmodel file')
    parser.add_argument('-t',
                        '--tokenizer_path',
                        type=str,
                        default="./configs/token_config",
                        help='path to the tokenizer file')
    parser.add_argument('-p',
                        '--processor_path',
                        type=str,
                        default="./configs/processor_config",
                        help='path to the processor file')
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default="./configs/config.json",
                        help='path to the model config file')
    parser.add_argument('-d', '--dev_id', type=int,
                        default=0, help='device ID to use')
    parser.add_argument('-g',
                        '--generation_mode',
                        type=str,
                        choices=["greedy", "penalty_sample"],
                        default="greedy",
                        help='mode for generating next token')
    parser.add_argument('-i',
                        '--input_paths',
                        type=str,
                        nargs="+",
                        default=["../datasets/videos/carvana_video.mp4",],
                        help='path to the video or images') 
    parser.add_argument('-ity',
                        '--input_type',
                        type=str,
                        choices=["image", "video"],
                        default="video",
                        help='input type') 
    parser.add_argument('-vc',
                        '--vision_preprocess_config',
                        type=str,
                        default="{}",
                        help='vision preprocess config') 
    parser.add_argument('-fsn',
                        '--frame_sample_num',
                        type=int,
                        default=1,
                        help='frame sampling interval') 
    args = parser.parse_args()
    main(args)
