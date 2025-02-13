#===----------------------------------------------------------------------===#
#
# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import gc
import inspect
import multiprocessing
import multiprocessing.shared_memory
import os
import signal
import sys
from typing import Any, Callable, Dict, List, Optional, Union

from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
import diffusers.utils.logging as logging
from diffusers.utils.torch_utils import randn_tensor
import numpy as np
import torch
from tqdm import tqdm
from transformers import CLIPTokenizer, T5TokenizerFast

import sophon.sail as sail

logger = logging.get_logger(__name__)

# define an unsupported exception when type(flux type, quantization type, chip type...) is not supported
class UnSupportedError(Exception):
    def __init__(self, type: str):
        self.type = type
        super().__init__(self._error_message())
    def _error_message(self):
        return f"{self.type} is not supported."

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

class StableDiffusion3Pipeline:
    # type config. 
    CHIP_TYPE = ('BM1690',)
    # number of blocks in each module
    CLIP_L_LAYER_NUM = 12
    CLIP_G_LAYER_NUM = 32
    T5_LAYER_NUM = 24
    MMDIT_LAYER_NUM = 24
    # device
    EXECUTION_DEVICE = torch.device("cpu")

    def __init__(
        self,
        # scheduler,
        # vae,
        # text_encoder,
        # tokenizer,
        # text_encoder_2,
        # tokenizer_2,
        # transformer,
    ):
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        self.vae_config_scaling_factor = 1
        self.vae_config_shift_factor = 0
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )
        self.default_sample_size = (
            self.transformer.config.sample_size
            if hasattr(self, "transformer") and self.transformer is not None
            else 128
        )
        self.transformer_config_in_channels = 16
        self.config_patch_size = 2
        self.out_channels = 16

    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    def from_models(
        self, 
        full_model_path: str = "../models",
        chip_type: str = "BM1690",
        device_ids: Union[int, List[int]] = 0, 
    ):
        r"""
        Load both model and bmodel files from full model directory, then allocate input/ouput memory.

        Args:
            full_model_path: direcotry of all model files.
            chip_type: product type.
            device_ids: TPU ID.
        """
        # 1. process and check parameters
        if not os.path.isdir(full_model_path):
            raise FileExistsError(f"No '{full_model_path}' directory exists.")

        chip_type = chip_type.upper()
        if chip_type not in self.CHIP_TYPE:
            raise UnSupportedError(chip_type)

        # 2. check tokenizer path
        #### clip_l
        tokenizer_path = os.path.join(full_model_path, "tokenizer")
        #### clip_g
        tokenizer_2_path = os.path.join(full_model_path, "tokenizer_2")
        #### t5
        tokenizer_3_path = os.path.join(full_model_path, "tokenizer_3")

        if not os.path.isdir(tokenizer_path):
            raise FileNotFoundError(f"No '{os.path.basename(tokenizer_path)}' directory found at {full_model_path}.")
        if not os.path.isdir(tokenizer_2_path):
            raise FileNotFoundError(f"No '{os.path.basename(tokenizer_2_path)}' directory found at {full_model_path}.")
        if not os.path.isdir(tokenizer_3_path):
            raise FileNotFoundError(f"No '{os.path.basename(tokenizer_3_path)}' directory found at {full_model_path}.")

        # 3. check bmodel files
        bmodel_path = os.path.join(full_model_path, chip_type)

        clip_l_path = os.path.join(full_model_path, chip_type, "clip_l.bmodel")
        if not os.path.isfile(clip_l_path):
            raise FileNotFoundError(f"No '{os.path.basename(clip_l_path)}' file found at {bmodel_path}.")

        clip_g_path = os.path.join(full_model_path, chip_type, "clip_g.bmodel")
        if not os.path.isfile(clip_g_path):
            raise FileNotFoundError(f"No '{os.path.basename(clip_g_path)}' file found at {bmodel_path}.")

        t5_path = os.path.join(full_model_path, chip_type, "t5.bmodel")
        if not os.path.isfile(t5_path):
            raise FileNotFoundError(f"No '{os.path.basename(t5_path)}' file found at {bmodel_path}.")

        vae_decoder_path = os.path.join(full_model_path, chip_type, "vae_decoder.bmodel")
        if not os.path.isfile(vae_decoder_path):
            raise FileNotFoundError(f"No '{os.path.basename(vae_decoder_path)}' file found at {bmodel_path}.")

        transformer_path = os.path.join(full_model_path, chip_type, f"mmdit.bmodel") 
        if not os.path.isfile(transformer_path):
            raise FileNotFoundError(f"No '{os.path.basename(transformer_path)}' file found at {bmodel_path}.")

        # 4. load tokenizers
        self.scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(tokenizer_2_path)
        self.tokenizer_3 = T5TokenizerFast.from_pretrained(tokenizer_3_path)

        # 5. load mmdit on device0
        self.device_ids = device_ids if type(device_ids) is list else [device_ids]
        if len(self.device_ids) == 2:
            if self.device_ids[0] == self.device_ids[1]:
                raise UnSupportedError("Two same device id")

        device_id0 = self.device_ids[0]

        transformer_engine = sail.nn.Engine(transformer_path, device_id0)

        # 6. load mmdit on second device and create shared mem if use two chips
        if len(self.device_ids) == 2:
            # prepare shared data
            mmdit_head_input_shapes = transformer_engine.get_input_shapes('mmdit_head', 0)
            mmdit_tail_output_shapes = transformer_engine.get_output_shapes('mmdit_tail', 0)

            self.shared_data_shapes =  mmdit_head_input_shapes + mmdit_tail_output_shapes
            self.shared_data_dtypes = [np.float32, np.float32, np.float32, np.int32, np.float32]

            shared_data = [np.zeros(shape, dtype=dtype) for shape, dtype in zip(self.shared_data_shapes, self.shared_data_dtypes)]
            shared_memories= []
            shared_memories_names = []
            for idx in range(len(shared_data)):
                shm = multiprocessing.shared_memory.SharedMemory(create=True, size=shared_data[idx].nbytes)
                shared_memories.append(shm)
                shared_memories_names.append(shm.name)
            self.event_start = multiprocessing.Event()
            self.event_done = multiprocessing.Event()
            self.process_child = multiprocessing.Process(target=self.child_process, args=(transformer_path, shared_memories_names))
            self.process_child.start()
            for idx in range(len(shared_memories)):
                setattr(self, f"shared_mem{idx}", shared_memories[idx]) 

        # 7. load text encoder and vae on chip0
        self.text_encoder =  sail.nn.Engine(clip_l_path, device_id0)
        self.text_encoder_2 = sail.nn.Engine(clip_g_path, device_id0)
        self.text_encoder_3 = sail.nn.Engine(t5_path, device_id0)
        self.vae_decoder = sail.nn.Engine(vae_decoder_path, device_id0)
        setattr(self, f"transformer_on_dev{device_id0}", transformer_engine)
        self.stream = sail.nn.Stream(device_id0)
    
    def child_process(self, transformer_path, shm_names):
        def handle_termination_signal(sig, frame):
            if hasattr(self, f"transformer_on_dev{self.device_ids[1]}"):
                delattr(self, f"transformer_on_dev{self.device_ids[1]}")
            for idx in range(len(shm_names)):
                getattr(self, f"shared_mem{idx}").close()
            gc.collect()
            sys.exit()

        signal.signal(signal.SIGTERM, handle_termination_signal)
        setattr(self, f"transformer_on_dev{self.device_ids[1]}", sail.nn.Engine(transformer_path, self.device_ids[1]))
        for idx in range(len(shm_names)):
            setattr(self, f"shared_mem{idx}", multiprocessing.shared_memory.SharedMemory(name=shm_names[idx], create=False))
        self.stream = sail.nn.Stream(self.device_ids[1])
        try:
            while True:
                self.event_start.wait()
                self.event_start.clear()
                shared_data = []
                for idx in range(len(shm_names)):
                    shared_data.append(torch.from_numpy(np.ndarray(self.shared_data_shapes[idx], self.shared_data_dtypes[idx], getattr(self, f"shared_mem{idx}").buf)))
                result = self.__mmdit_infer__(self.device_ids[1], *shared_data[:-1])
                np.copyto(shared_data[-1].numpy(), result.numpy())
                self.event_done.set()
        except Exception as e:
            print(f"child process error: {e}")
        finally:
            if hasattr(self, f"transformer_on_dev{self.device_ids[1]}"):
                delattr(self, f"transformer_on_dev{self.device_ids[1]}")
            gc.collect()

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 77,
        device: Optional[torch.device] = None,
        dtype: Optional[np.dtype] = None,
    ):
        r"""
        Get prompt embedding result of prompt via t5 text encoder

        Args:
            prompt: description of the wanted image, only support str(one prompt) now.
            num_images_per_prompt: only support one now.
            max_sequence_length: max token length.
            device: which device(where t5 is loaded) would receive the output of tokenizer, currently not in effect.
            dtype: set data type of result ,currently using float32.

        Returns:
            torch.Tensor: output of t5, shape is [1, 256, 4096].
        """
        device = device or self.EXECUTION_DEVICE
        dtype = dtype or np.float32

        # 1. get tokens of prompt
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if self.text_encoder_3 is None:
            return torch.zeros(
                (
                    batch_size * num_images_per_prompt,
                    self.tokenizer_max_length,
                    self.transformer.config.joint_attention_dim,
                ),
                device=device,
                dtype=dtype,
            )

        text_inputs = self.tokenizer_3(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_3(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_3.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        # 2. process of t5
        text_input_ids = text_input_ids.numpy().astype(np.int32)
        text_input_ids_tensor = sail.nn.Tensor(text_input_ids, sail.DataType.TPU_INT32, self.device_ids[0])
        t5_head_inputs = {0: text_input_ids_tensor}
        hidden_states_shape = self.text_encoder_3.get_output_shapes("t5_head", 0)[0]
        hidden_states_tensor = sail.nn.Tensor(hidden_states_shape, sail.DataType.TPU_FLOAT32, self.device_ids[0])
        t5_head_outputs = {0: hidden_states_tensor}

        ret = self.text_encoder_3.process(t5_head_inputs, t5_head_outputs, self.stream, "t5_head")

        t5_block_inputs = {0: hidden_states_tensor}
        t5_block_outputs = {0: hidden_states_tensor}
        for idx in range(self.T5_LAYER_NUM):
            ret = self.text_encoder_3.process(t5_block_inputs, t5_block_outputs, self.stream, f"t5_block_{idx}")
        t5_tail_inputs = {0: hidden_states_tensor}
        t5_tail_outputs = {0: hidden_states_tensor}
        ret = self.text_encoder_3.process(t5_tail_inputs, t5_tail_outputs, self.stream, "t5_tail")

        hidden_states_tensor.to_("host")
        prompt_embeds = hidden_states_tensor.asnumpy()
        prompt_embeds = torch.from_numpy(prompt_embeds)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        clip_skip: Optional[int] = None,
        clip_model_index: int = 0,
    ):
        r"""
        Get clip_pooling result of prompt

        Args:
            prompt: description of the wanted image, only support str(one prompt) now.
            num_images_per_prompt: only support one now.
            device: which device(where clip is loaded) would receive the output of tokenizer, currently not in effect.
            clip_skip: not in effect yet.
            clip_model_index: select the id of clip, 0 is clip_l, 1 is clip_g.

        Returns:
            torch.Tensor: output of clip_pooling, shape is [1, 768].
        """
        device = device or self.EXECUTION_DEVICE

        clip_tokenizers = [self.tokenizer, self.tokenizer_2]
        clip_text_encoders = [self.text_encoder, self.text_encoder_2]
        clip_layer_nums = [self.CLIP_L_LAYER_NUM, self.CLIP_G_LAYER_NUM]
        clip_types = ['clip_l', 'clip_g']

        tokenizer = clip_tokenizers[clip_model_index]
        text_encoder = clip_text_encoders[clip_model_index]
        block_nums = clip_layer_nums[clip_model_index]
        clip_type = clip_types[clip_model_index]

        if clip_skip is None:
            clip_skip = 0
        assert(clip_skip >= 0 and clip_skip <= block_nums - 2)
        # 1. get tokens of prompt
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )

        #### check whether the length of the prompt is too long
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )

        # 2. process of clip bmodel
        #### clip head process
        text_input_ids = text_input_ids.numpy().astype(np.int32 if clip_model_index == 1 else np.float32)
        text_input_ids_tensor = sail.nn.Tensor(text_input_ids, sail.DataType.TPU_INT32 if text_input_ids.dtype == np.int32 else sail.DataType.TPU_FLOAT32, self.device_ids[0])
        clip_head_inputs = {0: text_input_ids_tensor}
        hidden_states_shape = text_encoder.get_output_shapes(clip_type+"_head", 0)[0]
        hidden_states_tensor = sail.nn.Tensor(hidden_states_shape, sail.DataType.TPU_FLOAT32, self.device_ids[0])
        clip_head_outputs = {0: hidden_states_tensor}
        ret = text_encoder.process(clip_head_inputs, clip_head_outputs, self.stream, clip_type+"_head")

        #### clip block process
        clip_block_inputs = {0: hidden_states_tensor}
        clip_block_outputs= {0: hidden_states_tensor}
        for idx in range(block_nums):
            ret = text_encoder.process(clip_block_inputs, clip_block_outputs, self.stream, clip_type+"_block"+f"_{idx}")
            if idx == block_nums - (2 + clip_skip):
                hidden_states_tensor.to_("host")
                prompt_embeds = hidden_states_tensor.asnumpy()
                hidden_states_tensor.to_("device")

        clip_tail_inputs = {0: hidden_states_tensor, 1: text_input_ids_tensor}
        pooled_output = np.ndarray(shape=(hidden_states_shape[0], hidden_states_shape[2]), dtype=np.float32)
        pooled_output_tensor = sail.nn.Tensor(pooled_output, sail.DataType.TPU_FLOAT32, self.device_ids[0])
        clip_tail_outputs = {0: pooled_output_tensor}
        ret = text_encoder.process(clip_tail_inputs, clip_tail_outputs, self.stream, clip_type+"_tail")
        pooled_output_tensor.to_("host")

        # 3. convert to torch.Tensor
        prompt_embeds = torch.from_numpy(prompt_embeds)
        pooled_prompt_embeds = torch.from_numpy(pooled_output_tensor.asnumpy())

        _, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds, pooled_prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]],
        prompt_3: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        clip_skip: Optional[int] = None,
        max_sequence_length: int = 77,
        lora_scale: Optional[float] = None,
    ):
        r"""
        Encode prompt to text embedding, using clip and t5 in FLUX.1

        Args:
            prompt: prompt to be encoded, only support str(one prompt) now. 
            prompt_2: The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is 
                used in all text-encoders.
            device: currently not in effect.
            num_images_per_prompt: number of images that should be generated per prompt, only support one now.
            prompt_embeds:
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds:
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            lora_scale:
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded, not support currently.

        Returns:
            prompt_embeds: result of t5, shape [1, 512, 4096]
            pooled_promt_embeds: result of clip, shape [1, 77]
        """
        device = device or self.EXECUTION_DEVICE

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            prompt_3 = prompt_3 or prompt
            prompt_3 = [prompt_3] if isinstance(prompt_3, str) else prompt_3

            prompt_embed, pooled_prompt_embed = self._get_clip_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                clip_model_index=0,
            )
            prompt_2_embed, pooled_prompt_2_embed = self._get_clip_prompt_embeds(
                prompt=prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                clip_model_index=1,
            )
            clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)

            t5_prompt_embed = self._get_t5_prompt_embeds(
                prompt=prompt_3,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

            clip_prompt_embeds = torch.nn.functional.pad(
                clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
            )

            prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
            pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt
            negative_prompt_3 = negative_prompt_3 or negative_prompt

            # normalize str to list
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_2 = (
                batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
            )
            negative_prompt_3 = (
                batch_size * [negative_prompt_3] if isinstance(negative_prompt_3, str) else negative_prompt_3
            )

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embed, negative_pooled_prompt_embed = self._get_clip_prompt_embeds(
                negative_prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=None,
                clip_model_index=0,
            )
            negative_prompt_2_embed, negative_pooled_prompt_2_embed = self._get_clip_prompt_embeds(
                negative_prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=None,
                clip_model_index=1,
            )
            negative_clip_prompt_embeds = torch.cat([negative_prompt_embed, negative_prompt_2_embed], dim=-1)

            t5_negative_prompt_embed = self._get_t5_prompt_embeds(
                prompt=negative_prompt_3,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

            negative_clip_prompt_embeds = torch.nn.functional.pad(
                negative_clip_prompt_embeds,
                (0, t5_negative_prompt_embed.shape[-1] - negative_clip_prompt_embeds.shape[-1]),
            )

            negative_prompt_embeds = torch.cat([negative_clip_prompt_embeds, t5_negative_prompt_embed], dim=-2)
            negative_pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embed, negative_pooled_prompt_2_embed], dim=-1
            )

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def check_inputs(
        self,
        prompt,
        prompt_2,
        prompt_3,
        height,
        width,
        negative_prompt=None,
        negative_prompt_2=None,
        negative_prompt_3=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        # Check all inputs.
        if height != 512 or width != 512:
            raise ValueError(f"`height` and `width` have to be 512")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_3 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_3`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")
        elif prompt_3 is not None and (not isinstance(prompt_3, str) and not isinstance(prompt_3, list)):
            raise ValueError(f"`prompt_3` has to be of type `str` or `list` but is {type(prompt_3)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_2 is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_2`: {negative_prompt_2} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_3 is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_3`: {negative_prompt_3} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )

        if negative_prompt_embeds is not None and negative_pooled_prompt_embeds is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_pooled_prompt_embeds` also have to be passed. Make sure to generate `negative_pooled_prompt_embeds` from the same text encoder that was used to generate `negative_prompt_embeds`."
            )

        if max_sequence_length is not None and max_sequence_length > 256:
            raise ValueError(f"`max_sequence_length` cannot be greater than 256 but is {max_sequence_length}")

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def clip_skip(self):
        return self._clip_skip

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    def __mmdit_infer__(self, dev_id, latent_model_input, prompt_embed, pooled_prompt_embed, timestep):
        transformer = getattr(self, f'transformer_on_dev{dev_id}')
        latent_model_input = latent_model_input.numpy().astype(np.float32)
        timestep = timestep.numpy().astype(np.int32)
        prompt_embed = prompt_embed.numpy().astype(np.float32)
        pooled_prompt_embed = pooled_prompt_embed.numpy().astype(np.float32)

        latent_model_input_tensor = sail.nn.Tensor(latent_model_input, sail.DataType.TPU_FLOAT32, dev_id)
        timestep_tensor = sail.nn.Tensor(timestep, sail.DataType.TPU_INT32, dev_id)
        prompt_embed_tensor = sail.nn.Tensor(prompt_embed, sail.DataType.TPU_FLOAT32, dev_id)
        pooled_prompt_embed_tensor = sail.nn.Tensor(pooled_prompt_embed, sail.DataType.TPU_FLOAT32, dev_id)

        mmdit_head_inputs = {0: latent_model_input_tensor, 1: prompt_embed_tensor, 2: pooled_prompt_embed_tensor, 3: timestep_tensor}
        mmdit_head_output_shapes = transformer.get_output_shapes("mmdit_head", 0)
        hidden_states_tensor = sail.nn.Tensor(mmdit_head_output_shapes[0], sail.DataType.TPU_FLOAT32, dev_id)
        temb_tensor = sail.nn.Tensor(mmdit_head_output_shapes[1], sail.DataType.TPU_FLOAT32, dev_id)
        encoder_hidden_states_tensor = sail.nn.Tensor(mmdit_head_output_shapes[2], sail.DataType.TPU_FLOAT32, dev_id)
        mmdit_head_outputs = {
            0: hidden_states_tensor,
            1: temb_tensor,
            2: encoder_hidden_states_tensor,
        }
        ret = transformer.process(mmdit_head_inputs, mmdit_head_outputs, self.stream, "mmdit_head")
        temb_tensor.reshape([1, 1, 1536])

        #### mmdit block process
        mmdit_block_inputs = {0: hidden_states_tensor, 1: temb_tensor, 2: encoder_hidden_states_tensor}
        mmdit_block_outputs = {0: hidden_states_tensor}

        ret = transformer.process(mmdit_block_inputs, mmdit_block_outputs, self.stream, f"mmdit_block")
        
        #### mmdit tail process
        mmdit_tail_inputs = {0: hidden_states_tensor, 1: temb_tensor}
        mmdit_output_shape = transformer.get_output_shapes("mmdit_tail", 0)[0]
        latent = sail.nn.Tensor(mmdit_output_shape, sail.DataType.TPU_FLOAT32, dev_id)
        mmdit_tail_outputs = {0: latent}
        ret = transformer.process(mmdit_tail_inputs, mmdit_tail_outputs, self.stream, "mmdit_tail")
        latent.to_("host")
        output = torch.from_numpy(latent.asnumpy())
        return output



    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Union[str, List[str]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = False,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = None,
        max_sequence_length: int = 77,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt:
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead, only support str now.
            prompt_2:
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead, only support str now.
            prompt_3:
                The prompt or prompts to be sent to `tokenizer_3` and `text_encoder_3`. If not defined, `prompt` is
                will be used instead, only support str now.
            height: The height in pixels of the generated image, only support 512 now. 
            width: The width in pixels of the generated image. This is set to 512 by default for the best results.
            num_inference_steps:
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps:
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale:
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt:
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2:
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used instead
            negative_prompt_3:
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_3` and
                `text_encoder_3`. If not defined, `negative_prompt` is used instead
            num_images_per_prompt:
                The number of images to generate per prompt, only support one currently.
            generator: not support yet.
            latents:
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds:
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds:
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            output_type:
                The output format of the generate image. `PIL.Image.Image`
            return_dict: not support yet.
            joint_attention_kwargs: not support yet.
            callback_on_step_end: not support yet.
            callback_on_step_end_tensor_inputs: not support yet.
            max_sequence_length: Maximum sequence length to use with the `prompt`, support 77 now.

        Returns:
            PIL.Image.Image: generated image.
        """
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            prompt_3,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._clip_skip = clip_skip
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.EXECUTION_DEVICE

        # 3. set lora, not support now.
        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            clip_skip=self.clip_skip,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer_config_in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_height, latent_width = latent_model_input.shape[-2:]

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                if self.do_classifier_free_guidance:
                    if len(self.device_ids)==1:
                        noise_pred_uncond = self.__mmdit_infer__(self.device_ids[0], latent_model_input[0].unsqueeze(0), prompt_embeds[0].unsqueeze(0), pooled_prompt_embeds[0].unsqueeze(0), timestep[0].unsqueeze(0))
                    elif len(self.device_ids)==2:
                        shared_arrays = []
                        for idx in range(len(self.shared_data_shapes)):
                            shared_arrays.append(np.ndarray(self.shared_data_shapes[idx], self.shared_data_dtypes[idx], getattr(self, f"shared_mem{idx}").buf))
                        shared_data = [latent_model_input[0].unsqueeze(0), prompt_embeds[0].unsqueeze(0), pooled_prompt_embeds[0].unsqueeze(0), timestep[0].unsqueeze(0)]
                        for idx in range(len(shared_data)):
                            shared_arrays[idx][:] = shared_data[idx].numpy()
                        self.event_start.set()
                    noise_pred_text = self.__mmdit_infer__(self.device_ids[0], latent_model_input[1].unsqueeze(0), prompt_embeds[1].unsqueeze(0), pooled_prompt_embeds[1].unsqueeze(0), timestep[1].unsqueeze(0))
                    if len(self.device_ids)==2:
                        self.event_done.wait()
                        noise_pred_uncond = torch.from_numpy(np.ndarray(self.shared_data_shapes[-1],self.shared_data_dtypes[-1],getattr(self, f"shared_mem{len(self.shared_data_shapes)-1}").buf))
                        self.event_done.clear()
                    noise_pred = torch.cat([noise_pred_uncond, noise_pred_text])
                    noise_pred = noise_pred.reshape(
                        shape=(noise_pred.shape[0], latent_height//self.config_patch_size, latent_width//self.config_patch_size, self.config_patch_size, self.config_patch_size, self.out_channels)
                    )
                    noise_pred = torch.einsum("nhwpqc->nchpwq", noise_pred)
                    noise_pred = noise_pred.reshape(
                        shape=(noise_pred.shape[0], self.out_channels, latent_height, latent_width)
                    )
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                else:
                    noise_pred = self.__mmdit_infer__(self.device_ids[0], latent_model_input, prompt_embeds, pooled_prompt_embeds, timestep)
                    noise_pred = noise_pred.reshape(
                        shape=(noise_pred.shape[0], latent_height // self.config_patch_size, latent_width // self.config_patch_size, self.config_patch_size, self.config_patch_size, self.out_channels)
                    )
                    noise_pred = torch.einsum("nhwpqc->nchpwq", noise_pred)
                    noise_pred = noise_pred.reshape(
                        shape=(noise_pred.shape[0], self.out_channels, latent_height, latent_width)
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                # if XLA_AVAILABLE:
                #     xm.mark_step()

        if output_type == "latent":
            image = latents

        else:
            #### combined to vae bmodel
            latents = (latents / self.vae_config_scaling_factor) + self.vae_config_shift_factor
            #### process of vae
            latents = latents.numpy().astype(np.float32)
            latents_tensor = sail.nn.Tensor(latents, sail.DataType.TPU_FLOAT32, self.device_ids[0])
            vae_inputs = {0: latents_tensor}
            vae_outputs_shape = self.vae_decoder.get_output_shapes("vae_decoder", 0)[0]
            vae_outputs = {0: sail.nn.Tensor(vae_outputs_shape, sail.DataType.TPU_FLOAT32, self.device_ids[0])}
            ret = self.vae_decoder.process(vae_inputs, vae_outputs, self.stream, "vae_decoder")
            vae_outputs[0].to_("host")
            image = vae_outputs[0].asnumpy()
            image = torch.from_numpy(image)
            image = self.image_processor.postprocess(image, output_type=output_type)

        if not return_dict:
            return (image,)

        return image

    def __del__(self):
        if hasattr(self, "process_child"):
            if self.process_child.is_alive():
                self.process_child.terminate()
                self.process_child.join(timeout=3)
        try:
            for idx in range(len(self.shared_data_shapes)):
                shared_mem = getattr(self, f"shared_mem{idx}", None)
                shared_mem.close()
                shared_mem.unlink()
        except:
            pass
        return
