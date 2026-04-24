import time
import argparse
from sophon import sail
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLConfig, BatchFeature, GenerationConfig
from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessorKwargs
from vision_process import process_vision_info
import json
import torch
import numpy as np
import torch.nn.functional as F
import copy
import SILK2.Tools.logger as Logger
import readline

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

class Qwen2_5_VLInputProcessor:
    """ overwrite <class 'transformers.models.qwen2_5_vl.processing_qwen2_vl.Qwen2_5_VLProcessor'>.__call__ function
            to use exsiting image_grid_thw or video_grid_thw
    """
    def __init__(self, processor_path, trust_remote_code=True, **kwargs):
        self.version = "1.0.0"
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
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            videos (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of videos to be prepared. Each video can be a 4D NumPy array or PyTorch
                tensor, or a nested list of 3D frames. Both channels-first and channels-last formats are supported.
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
            - **second_per_grid_ts** -- List of video seconds per time grid. Returned when `videos` is not `None`.
        """
        output_kwargs = self.processor._merge_kwargs(
            Qwen2_5_VLProcessorKwargs,
            tokenizer_init_kwargs=self.processor.tokenizer.init_kwargs,
            **kwargs,
        )
        if images is not None:
            image_inputs = self.processor.image_processor(images=images, videos=None, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = {}

        if videos is not None:
            videos_inputs = self.processor.image_processor(images=None, videos=videos, **output_kwargs["images_kwargs"])
            video_grid_thw = videos_inputs["video_grid_thw"]

            fps = output_kwargs["videos_kwargs"].pop("fps", 2.0)
            if isinstance(fps, (int, float)):
                second_per_grid_ts = [self.processor.image_processor.temporal_patch_size / fps] * len(video_grid_thw)
            elif hasattr(fps, "__len__") and len(fps) == len(video_grid_thw):
                second_per_grid_ts = [self.processor.image_processor.temporal_patch_size / tmp for tmp in fps]
            else:
                raise ValueError(
                    f"The length of fps ({len(fps) if hasattr(fps, '__len__') else fps}) must be equal to the length of video_grid_thw ({len(video_grid_thw)}) or fps should be a single number."
                )
            videos_inputs.update({"second_per_grid_ts": second_per_grid_ts})
        else:
            videos_inputs = {}

        if not isinstance(text, list):
            text = [text]

        if image_grid_thw is not None:
            merge_length = self.processor.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while "<|image_pad|>" in text[i]:
                    text[i] = text[i].replace(
                        "<|image_pad|>",
                        "<|placeholder|>" * (image_grid_thw[index].prod() // merge_length),
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", "<|image_pad|>")

        if video_grid_thw is not None:
            merge_length = self.processor.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while "<|video_pad|>" in text[i]:
                    text[i] = text[i].replace(
                        "<|video_pad|>",
                        "<|placeholder|>" * (video_grid_thw[index].prod() // merge_length),
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", "<|video_pad|>")

        _ = output_kwargs["text_kwargs"].pop("padding_side", None)
        text_inputs = self.processor.tokenizer(text, **output_kwargs["text_kwargs"])

        return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs})

    def compute_ids_length_exclude_vision(self, text, **kwargs):
        output_kwargs = self.processor._merge_kwargs(
            Qwen2_5_VLProcessorKwargs,
            tokenizer_init_kwargs=self.processor.tokenizer.init_kwargs,
            **kwargs,
        )
        text = copy.deepcopy(text)
        if not isinstance(text, list):
            text = [text]
        for indx in range(len(text)):
            text[indx] = text[indx].replace("<|image_pad|>", "")
            text[indx] = text[indx].replace("<|video_pad|>", "")
        return self.processor.tokenizer(text, **output_kwargs["text_kwargs"])["input_ids"].shape[1]

    def apply_chat_template(self, msg, tokenize=False, add_generation_prompt=True, **kwargs):
        return self.processor.apply_chat_template(msg, tokenize=tokenize, add_generation_prompt=add_generation_prompt, **kwargs)


class Qwen2_5_VL():

    def __init__(self, **kwargs):
        self.version = "1.0.0"
        # init logger
        self.logger = Logger.Log("Qwen2.5VL", kwargs.get("log_level", "INFO"))

        self.logger.info(f"{Logger.file_lineno()} loading model {kwargs['bmodel_path']} to dev:{kwargs.get('dev_id', 0)}")
        st = time.time()
        # devid
        self.dev_id = kwargs.get("dev_id", 0)
        self.handle = sail.Handle(self.dev_id)
        # LLM bmodel inference on A2 may cause device memory increasing, which should be solved by set bmrt_set_flags(BM_RUNTIME_SHARE_MEM).
        self.net = sail.EngineLLM(kwargs["bmodel_path"], sail.BmrtFlag.BM_RUNTIME_SHARE_MEM, [self.dev_id])
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
        self.first_hidden_states_input_shape = self.net.get_input_shape("block_0", 0)
        self.vit_hidden_states_input_shape = self.net.get_input_shape("vit", 0)
        self.vit_pos_ids_input_shape = self.net.get_input_shape("vit", 1)
        self.vit_attention_mask_input_shape = self.net.get_input_shape("vit", 2)
        self.reverse_index_input_shape = self.net.get_input_shape("vit", 4)
        _, self.seq_len, self.hidden_size = self.first_hidden_states_input_shape
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
        self.do_sample = kwargs.get("do_sample", False)
        self.is_dynamic = self.net.get_is_dynamic("block_0")
        # initialize net name
        self.name_embed = "embedding"
        self.name_embed_cache = "embedding_cache"
        self.name_lm = "lm_head"
        self.name_blocks = ["block_"+str(i) for i in range(self.num_layers)]
        self.name_blocks_cache = ["block_cache_"+str(i) for i in range(self.num_layers)]
        self.name_vit = "vit"
        self.greedy = "greedy_head"
        self.sample = "sample_head"
        if kwargs.get("do_sample", False) and self.sample in self.graph_names:
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
        config_dir = kwargs["config_path"]
        self.processor = Qwen2_5_VLInputProcessor(config_dir,
                                                       trust_remote_code=True,
                                                       max_pixels=self.vision_seq_len * 14 * 14,
                                                       min_pixels=256 * 28 * 28
                                                       )
        self.tokenizer = AutoTokenizer.from_pretrained(config_dir,
                                                       trust_remote_code=True)
        with open(str(config_dir + '/config.json'), 'r') as f:
            self.config = json.load(f)
        self.loaded_config = Qwen2_5_VLConfig(**self.config)
        self.ID_END = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        self.ID_IM_END = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.ID_IMAGE_PAD = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.ID_VIDEO_PAD = self.tokenizer.convert_tokens_to_ids("<|video_pad|>")
        self.ID_VISION_START = self.tokenizer.convert_tokens_to_ids('<|vision_start|>')
        self.spatial_merge_size = self.loaded_config.vision_config.spatial_merge_size
        self.spatial_merge_unit = self.spatial_merge_size**2
        self.token_len = 0
        self.tokens_per_second = 2
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

    def get_rope_index(self, input_ids: torch.LongTensor, grid_thw: torch.LongTensor,
                       pad_id: int) -> torch.Tensor:
        total_input_ids = input_ids
        attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index = 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums = 0
            vision_start_indices = torch.argwhere(input_ids == self.ID_VISION_START).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == pad_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images = image_nums
            for _ in range(image_nums):
                if pad_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(pad_id, st)
                else:
                    ed_image = len(input_tokens) + 1

                t, h, w = (
                    grid_thw[image_index][0],
                    grid_thw[image_index][1],
                    grid_thw[image_index][2],
                )
                second_per_grid_t = 0
                image_index += 1
                remain_images -= 1
                ed = ed_image

                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // self.spatial_merge_size,
                    w.item() // self.spatial_merge_size,
                )
                text_len = ed - st

                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                time_tensor = expanded_range * second_per_grid_t * self.tokens_per_second

                time_tensor_long = time_tensor.long()
                t_index = time_tensor_long.flatten()

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
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
        return position_ids
        
    def preprocess(
        self, 
        messages, 
        image_grid_thw=None, 
        video_grid_thw=None
    ):
        # init prompt
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, add_vision_id=False) # add_vision_id=False) has higher precision
            for msg in messages
        ]

        # compute max vision seq length
        vision_seq_max_ratio = self.vision_seq_max_ratio
        vision_max_output_seq_length = self.seq_len * vision_seq_max_ratio - self.processor.compute_ids_length_exclude_vision(text=texts, padding=True, return_tensors="pt")
        vision_max_input_seq_length = (self.spatial_merge_size ** 2) * vision_max_output_seq_length
        real_vision_max_input_shape = [*self.vit_hidden_states_input_shape]
        real_vision_max_input_shape[0] = min(real_vision_max_input_shape[0], vision_max_input_seq_length) # consider both llm and vision seq length

        if image_grid_thw is None and video_grid_thw is None:
            image_inputs, video_inputs = process_vision_info(self.handle, messages, \
                max_vision_shape=real_vision_max_input_shape, merge_size=self.loaded_config.vision_config.spatial_merge_size, logger=self.logger)
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

        return inputs

    def rot_pos(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        return pos_ids
    
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

        input_ids = np.zeros((tokens.shape[0], self.seq_len), dtype=type_convert(self.input_tensors[self.name_embed][0].dtype()))
        input_ids[:, :min(self.seq_len, tokens.shape[1])] = tokens
        self.input_tensors[self.name_embed][0].update_data(input_ids)
        self.net.process(self.name_embed, self.input_tensors[self.name_embed], self.output_tensors[self.name_embed])

    def forward_vit(self, pixel_values, position_ids, full_attn_mask, window_attn_mask, grid_thw, reverse_index, vit_offset):
        t, h, w = grid_thw.squeeze(0).tolist()
        volume = t * h * w
        assert full_attn_mask.numel() == volume ** 2
        assert window_attn_mask.numel() == volume ** 2
        assert pixel_values.numel() == volume * self.vit_hidden_states_input_shape[1]
        assert position_ids.numel() == volume * 2
        assert reverse_index.numel() == volume // 4

        pixel_values = pixel_values.numpy().astype(type_convert(self.input_tensors[self.name_vit][0].dtype()))
        pixel_values_prefill = np.zeros(self.vit_hidden_states_input_shape, dtype=type_convert(self.input_tensors[self.name_vit][0].dtype()))
        pixel_values_prefill[:pixel_values.shape[0],:] = pixel_values

        position_ids = position_ids.numpy().astype(type_convert(self.input_tensors[self.name_vit][1].dtype()))
        pos_ids_prefill = np.zeros(self.vit_pos_ids_input_shape, dtype=type_convert(self.input_tensors[self.name_vit][1].dtype()))
        pos_ids_prefill[:position_ids.shape[0],:] = position_ids
        
        reverse_index = reverse_index.numpy().astype(type_convert(self.input_tensors[self.name_vit][4].dtype()))
        reverse_index_prefill = np.zeros(self.reverse_index_input_shape, dtype=type_convert(self.input_tensors[self.name_vit][4].dtype()))
        reverse_index_prefill[:reverse_index.shape[0]] = reverse_index

        self.input_tensors[self.name_vit][0].update_data(pixel_values_prefill)
        self.input_tensors[self.name_vit][1].update_data(pos_ids_prefill)
        self.input_tensors[self.name_vit][4].update_data(reverse_index_prefill)

        if full_attn_mask.numel() == self.vision_seq_len ** 2:
            full_attn_mask = full_attn_mask.numpy().astype(type_convert(self.input_tensors[self.name_vit][2].dtype()))
            window_attn_mask = window_attn_mask.numpy().astype(type_convert(self.input_tensors[self.name_vit][3].dtype()))
            self.input_tensors[self.name_vit][2].update_data(full_attn_mask)
            self.input_tensors[self.name_vit][3].update_data(window_attn_mask)
        else:
            mask_full = torch.full(
                (1, 1, self.vision_seq_len, self.vision_seq_len),
                -10000.0,
                dtype=full_attn_mask.dtype,
                device=full_attn_mask.device
            )
            mask_window = torch.full(
                (1, 1, self.vision_seq_len, self.vision_seq_len),
                -10000.0,
                dtype=window_attn_mask.dtype,
                device=window_attn_mask.device
            )

            mask_full[:, :, :volume, :volume] = full_attn_mask[:, :, :volume, :volume]
            mask_window[:, :, :volume, :volume] = window_attn_mask[:, :, :volume, :volume]
            self.input_tensors[self.name_vit][2].update_data(mask_full)
            self.input_tensors[self.name_vit][3].update_data(mask_window)

        self.net.process(self.name_vit, self.input_tensors[self.name_vit], self.output_tensors[self.name_vit])

        dst_offset = vit_offset * self.hidden_size
        vit_size   = (volume // 4)   * self.hidden_size
        vision_embeds = self.output_tensors[self.name_vit][0].asnumpy().reshape(-1)
        token_embeds = self.output_tensors[self.name_embed][0].asnumpy().reshape(-1)
        token_embeds[dst_offset : dst_offset + vit_size] = vision_embeds[:vit_size]
        token_embeds = token_embeds.reshape(self.output_tensors[self.name_embed][0].asnumpy().shape)

        self.output_tensors[self.name_embed][0].update_data(token_embeds)

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
            generated_tokens = np.zeros([1, self.seq_len], type_convert(self.input_tensors[self.sample][1].dtype()))
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
            attention_mask = [ATTENTION_MASK] * (self.seq_len * self.seq_len)
            for i in range(self.token_len):
                for j in range(self.token_len):
                    if j <= i:
                        attention_mask[i * self.seq_len + j] = 0
            attention_mask = np.array(attention_mask, dtype=type_convert(self.input_tensors[self.name_blocks[0]][2].dtype())).reshape(self.input_tensors[self.name_blocks[0]][2].shape())

            position_ids_pad = [0] * (3 * self.seq_len)
            ori_length = len(position_ids) // 3
            for i in range(3):
                ori_offset = i * ori_length
                dst_offset = i * self.seq_len
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

        # linear process
        self.step = self.token_len
        offset_bytes = self.first_hidden_states_output.size() // self.seq_len
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
        position_ids = np.array([position_id]*3, dtype=np.int32).reshape(3, 1)
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

    def generate_message(self, history_messages, cur_data, cur_role):
        """ accept current question and vision info, will combine with history_messages

        params:
            history_messages: list, element is dict, qwen2-vl messages' format
            cur_data: list or str. if list, element is dict. qwen2-vl messages' `content` format
            cur_role: str, current role, support [`user`, `system`, `assistant`]

        return:
            messages: messages input to model, list, element is dict, qwen2-vl messages' format
        """
        self.logger.debug(f"{Logger.file_lineno()} receive data history: {history_messages}, current data: {cur_data}, current role: {cur_role}, which need convert to qwen2-vl messages format")

        cur_data = copy.deepcopy(cur_data)
        assert cur_role in ["system", "assistant", "user"]
        assert isinstance(history_messages, list)
        messages = [
            {
                "role": cur_role,
                "content": [
                ],
            }
        ]

        media_type = None
        if isinstance(cur_data, str):
            messages[0]["content"] = cur_data
        elif isinstance(cur_data, list):
            for ele in cur_data:
                assert ele["type"] in ["image_url", "video_url", "text"], \
                        f"get {ele['type']}, expect: image_url/video_url/text"
                if ele["type"] == "image_url":
                    assert isinstance(ele[ele["type"]], dict)
                    ele["image"] = ele[ele["type"]]["url"]
                    del ele[ele["type"]]
                    ele["type"] = "image"
                    media_type = "image"
                elif ele["type"] == "video_url":
                    assert isinstance(ele[ele["type"]], dict)
                    ele["video"] = ele[ele["type"]]["url"]
                    del ele[ele["type"]]
                    ele["type"] = "video"
                    media_type = "video"
                elif ele["type"] == "text":
                    assert isinstance(ele[ele["type"]], str)
                messages[0]["content"].append(ele)
        else:
            raise TypeError(f"cur_data only support list or str type, qwen2-vl messages' `content` format")

        if len(messages[0]["content"]) == 0:
            self.logger.debug(f"{Logger.file_lineno()} get model messages input {history_messages}")
            return history_messages
        history_messages.extend(messages)
        messages = history_messages
        self.logger.debug(f"{Logger.file_lineno()} get model messages input {messages}")
        return messages, media_type
    
    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = 4

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // 2,
                grid_w // 2,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
                grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)
        return window_index, cu_window_seqlens

    def vit_process_image(self,vit_offset, grid_thw, hidden_states):
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(cu_window_seqlens, dtype=torch.int32)
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
        seq_len, _ = hidden_states.shape
        # reorder hidden_states
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit,
                                              self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        # reorder position_ids
        position_ids = self.rot_pos(grid_thw)
        position_ids = position_ids.reshape(seq_len // self.spatial_merge_unit,
                                            self.spatial_merge_unit, -1)
        position_ids = position_ids[window_index, :, :]
        position_ids = position_ids.reshape(seq_len, -1)
        # cu_seqlens
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2],
                                             grid_thw[:, 0]).cumsum(
                                                 dim=0,
                                                 dtype=torch.int32,
                                             )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        full_mask = self.get_attn_mask(seq_len, cu_seqlens)
        window_mask = self.get_attn_mask(seq_len, cu_window_seqlens)
        reverse_indices = torch.argsort(window_index)
        self.forward_vit(hidden_states,
                        position_ids,
                        full_mask,
                        window_mask,
                        grid_thw,
                        reverse_indices,
                        vit_offset)

    def vit_process_video(self, vit_offset, video_grid_thw, pixel_values_videos):
        t, h, w = video_grid_thw.flatten().tolist()
        per_t = self.vision_seq_len // (h * w)
        t_list = []
        if per_t >= t:
            t_list = [t]
        else:
            t_list = [per_t] * (t // per_t) + ([t % per_t] if t % per_t else [])
        t_offset = 0
        for t_i in t_list:
            grid_thw = torch.tensor([[t_i, h, w]], dtype=torch.int32)
            hidden_states = pixel_values_videos[(t_offset * h * w):((t_offset + t_i) * h *
                                                                           w), :]
            self.vit_process_image(vit_offset, grid_thw, hidden_states)
            seq_len, _ = hidden_states.shape
            vit_offset += seq_len // 4
            t_offset += t_i

    def update_embeddings(self, inputs, vision_embeds, pad):
        emb_buf = self.output_tensors[self.name_embed][0].asnumpy()
        emb_np  = emb_buf.reshape(-1, self.hidden_size)

        token_ids = inputs.input_ids.numpy().reshape(-1)
        pad_pos   = np.where(token_ids == pad)[0]

        mask = np.zeros((emb_np.shape[0],), dtype=bool)
        mask[pad_pos] = True

        vision_embeds = vision_embeds.reshape(-1, self.hidden_size)
        emb_np[mask] = vision_embeds[mask]

        new_buf = emb_np.reshape(emb_buf.shape)
        self.output_tensors[self.name_embed][0].update_data(new_buf)

    def is_end_with_reason(self, token):
        return token in [self.ID_IM_END, self.ID_END
                                ], self.step >= self.seq_len

    def decode(self, tokens, **kwargs):
        return self.tokenizer.decode(tokens, **kwargs)

def main(args):
    history_messages = []
    model = Qwen2_5_VL(dev_id=args.dev_id, bmodel_path=args.bmodel_path, log_level=args.log_level, config_path=args.config_path, do_sample=args.do_sample)
    media_type = None
    vit_offset = None
    vision_embeds = []
    video_grid_thw = None
    image_grid_thw = None
    pixel_values_images = None
    pixel_values_videos = None
    vision_inputs = args.vision_inputs
    if len(vision_inputs) > 0:
        vision_inputs = json.loads(vision_inputs)
    else:
        vision_inputs = []

    print(
        "\n================================================================="
            "\n1. If you want to quit, please enter one of [q]"
            "\n2. If you want to clear history, please enter one of [c]"
            "\n=================================================================")
    while True:
        text = input("\nQuestion: ").strip()
        if len(text) == 0:
            print("Please input content")
            continue
        elif text == "q":
            print("accept 'q', exit")
            break
        elif text == "c":
            print("accept 'c', clear history")
            history_messages = []
            continue

        first_start = time.time()
        cur_media_type = None
        if len(history_messages) == 0 and len(vision_inputs) > 0:
            cur_data = copy.deepcopy(vision_inputs)
            cur_data.append({"type": "text", "text": text})
            messages, cur_media_type = model.generate_message(history_messages, cur_data, "user")
        else:
            messages, cur_media_type = model.generate_message(history_messages, text, "user")
        messages = [messages]
        media_type = cur_media_type if media_type is None else media_type

        # preprocess text and images/video, get model inputs
        inputs = model.preprocess(messages, image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw)
        if image_grid_thw is None and video_grid_thw is None:
            image_grid_thw = inputs.image_grid_thw if "image_grid_thw" in inputs else None
            video_grid_thw = inputs.video_grid_thw if "video_grid_thw" in inputs else None
            pixel_values_images = inputs.pixel_values if "pixel_values" in inputs else None
            pixel_values_videos = inputs.pixel_values_videos if "pixel_values_videos" in inputs else None

        token_len = inputs.input_ids.numel()
        model.forward_embed(inputs.input_ids.numpy())
        position_ids = np.tile(np.arange(token_len), 3)
        max_posid = token_len - 1

        if media_type == "image":
            if len(vision_embeds) == 0:
                vit_token_list = torch.where(inputs.input_ids == model.ID_IMAGE_PAD)[1].tolist()
                vit_offset = vit_token_list[0]
                model.vit_process_image(vit_offset, image_grid_thw, pixel_values_images)
                vision_embeds = model.output_tensors[model.name_embed][0].asnumpy()
            position_ids = model.get_rope_index(inputs.input_ids, image_grid_thw,
                                                model.ID_IMAGE_PAD)
            max_posid = int(position_ids.max())
            position_ids = position_ids.numpy()

            # 把<|image_pad|>替换成缓存好的视觉embedding
            model.update_embeddings(inputs, vision_embeds, model.ID_IMAGE_PAD)
        elif media_type == "video":
            if len(vision_embeds) == 0:
                vit_token_list = torch.where(inputs.input_ids == model.ID_VIDEO_PAD)[1].tolist()
                vit_offset = vit_token_list[0]
                model.vit_process_video(vit_offset, video_grid_thw, pixel_values_videos)
                vision_embeds = model.output_tensors[model.name_embed][0].asnumpy()
            position_ids = model.get_rope_index(inputs.input_ids, video_grid_thw,
                                                model.ID_VIDEO_PAD)
            max_posid = int(position_ids.max())
            position_ids = position_ids.numpy()

            # 把<|video_pad|>替换成缓存好的视觉embedding
            model.update_embeddings(inputs, vision_embeds, model.ID_VIDEO_PAD)

        # Chat
        print("\nAnswer: ", end = '')
        token = model.forward_first(position_ids)
        
        first_end = time.time()
        tok_num = 0
        # Following tokens
        full_word_tokens = []
        text = ""
        while not (model.is_end_with_reason(token)[0] or model.is_end_with_reason(token)[1]):
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
            max_posid += 1

            token = model.forward_next(max_posid)
            tok_num += 1
        next_end = time.time()
        first_duration = first_end - first_start
        next_duration = next_end - first_end
        tps = tok_num / next_duration
        print(f"\nFTL: {first_duration:.3f} s")
        print(f"TPS: {tps:.3f} token/s")
        history_messages.append({
            "role": "assistant",
            "content": text,
        })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--bmodel_path',
                        type=str,
                        default="../models/BM1684X/qwen2.5-vl-3b-instruct-awq_w4bf16_seq2048_bm1684x_1dev_20250428_143625.bmodel",
                        help='path to the bmodel file')
    parser.add_argument('-c',
                        '--config_path',
                        type=str,
                        default="./configs",
                        help='path to the model config file')
    parser.add_argument('-d', '--dev_id', type=int,
                        default=0, help='device ID to use')
    parser.add_argument('-vi',
                        '--vision_inputs',
                        type=str,
                        default="[{\"type\":\"image_url\", \"image_url\":{\"url\":\"../datasets/images/bookstore_17_02_flickr.jpg\"}, \
                                    \"max_side\":420, \"resize_type\": \"INTER_LINEAR\"}]",
                        help='path to the video or images and preprocess params, json format') 
    parser.add_argument('-ll',
                        '--log_level',
                        type=str,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        default="INFO",
                        help='log level, default: INFO, option[DEBUG, INFO, WARNING, ERROR]')
    parser.add_argument('--do_sample', 
                        action='store_true', 
                        help="if set, generate tokens by sample parameters")
    args = parser.parse_args()
    main(args)