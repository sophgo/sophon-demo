# FLUX.1 [dev] Non-Commercial License
#===----------------------------------------------------------------------===#
#
# Copyright (C) 2024 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import inspect
from functools import reduce
import operator
import os
import random
from typing import Any, Callable, Dict, List, Optional, Union

from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor
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

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

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

class FluxPipeline:
    # type config. 
    FLUX_TYPE = ('dev', 'schnell', )
    QUANT_DTYPE = ("w4bf16", "bf16", )
    CHIP_TYPE = ('BM1684X', )
    #### clip and vae are loaded on device0, t5 is loaded on device2
    MULTI_DEVICE_ALLOCATION = {'clip': 0, 'vae': 0,'t5': 2, }
    CLIP_LAYER_NUM = 12
    T5_LAYER_NUM = 24
    #### splitting transformer blocks
    MM_TRANS_END_FLAG_ON_DEV0 = 12
    MM_TRANS_LAYERS = 19
    SINGLE_TRANS_END_FLAG_ON_DEV1 = 27
    SINGLE_TRANS_LAYERS = 38
    #### scheduler config
    SCHEDULE_CONFIG = {
    "schnell": {
        "base_image_seq_len": 256,
        "base_shift": 0.5,
        "max_image_seq_len": 4096,
        "max_shift": 1.15,
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "use_dynamic_shifting": False,
    },
    "dev": {
        "base_image_seq_len": 256,
        "base_shift": 0.5,
        "max_image_seq_len": 4096,
        "max_shift": 1.15,
        "num_train_timesteps": 1000,
        "shift": 3.0,
        "use_dynamic_shifting": True
    },
    }
    EXECUTION_DEVICE = torch.device("cpu")
    MULTI_DEVICES_NUM = 3
    IMAGE_ROTARY_EMB_SHAPE = [1, 4608, 1, 64, 2, 2]

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
        #### Config for sub modules, hyperparameters come from the original config.json
        self.default_sample_size = 64
        self.vae_scale_factor = 16
        #### process before vae has been integrated into vae_decoder.bmodel, so the two parameters below are not used
        self.vae_config_scaling_factor = 0.3611
        self.vae_config_shift_factor = 0.1159

        self.transformer_config_in_channels = 64
        self.transformer_config_guidance_embeds = True

        self.tokenizer_max_length = 77
        self.tokenizer_2_max_length = 512

        self.default_sample_size = 64

        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

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
        chip_type: str = "BM1684X",
        device_ids: Union[int, List[int]] = [0, 1, 2], 
        quant_dtype: str = "bf16",
        flux_type: str = "dev",
        use_tiny_vae: bool = False
    ):
        r"""
        Load both model and bmodel files from full model directory, then allocate input/ouput memory.

        Args:
            model_path: direcotry of all model files.
            chip_type: product type.
            device_ids: specify device number, 1 device for w4bf16 quant dtype, MULTI_DEVICES_NUM devices for bf16.
            quant_dtype: w4bf16(INT4 weight, BF16 activation), bf16(BF16 weight, BF16 activatetion). 
            use_tiny_vae: use taef1 if it is True, else use original vae 
        """
        # 1. process and check parameters
        if not os.path.isdir(full_model_path):
            raise FileExistsError(f"No '{full_model_path}' directory exists.")

        chip_type = chip_type.upper()
        if chip_type not in self.CHIP_TYPE:
            raise UnSupportedError(chip_type)

        device_ids = device_ids if isinstance(device_ids, list) else [device_ids]

        quant_dtype = quant_dtype.lower()
        if quant_dtype not in self.QUANT_DTYPE:
            raise UnSupportedError(quant_dtype)
        if quant_dtype == "bf16" and len(device_ids) != self.MULTI_DEVICES_NUM:
            raise Exception("Need three devices.")
        self.quant_dtype = quant_dtype

        flux_type = flux_type.lower()
        if flux_type not in self.FLUX_TYPE:
            raise UnSupportedError(flux_type)
        self.flux_type = flux_type

        # 2. check normal model files, not bmodel files
        #### check tokenizer path
        tokenizer_path = os.path.join(full_model_path, "tokenizer")
        tokenizer_2_path = os.path.join(full_model_path, "tokenizer_2")
        if not os.path.isdir(tokenizer_path):
            raise FileNotFoundError(f"No '{os.path.basename(tokenizer_path)}' directory found at {full_model_path}.")
        if not os.path.isdir(tokenizer_2_path):
            raise FileNotFoundError(f"No '{os.path.basename(tokenizer_2_path)}' directory found at {full_model_path}.")

        #### check rotary embedding model
        rotary_emb_path = os.path.join(full_model_path, "ids_emb.pt")
        if not os.path.isfile(rotary_emb_path):
            raise FileNotFoundError(f"No '{os.path.basename(rotary_emb_path)}' file found at {full_model_path}.")

        # 2. check bmodel files
        bmodel_path = os.path.join(full_model_path, chip_type)
        quant_dtype = quant_dtype.lower()

        clip_path = os.path.join(full_model_path, chip_type, "clip.bmodel")
        if not os.path.isfile(clip_path):
            raise FileNotFoundError(f"No '{os.path.basename(clip_path)}' file found at {bmodel_path}.")

        #### t5 must be w4bf16, because it has 9 billion parameters
        t5_path = os.path.join(full_model_path, chip_type, "w4bf16_t5.bmodel")
        if not os.path.isfile(t5_path):
            raise FileNotFoundError(f"No '{os.path.basename(t5_path)}' file found at {bmodel_path}.")

        vae_decoder_path = os.path.join(full_model_path, chip_type, "vae_decoder_bf16.bmodel" if use_tiny_vae is False else "tiny_vae_decoder_bf16.bmodel")
        if not os.path.isfile(vae_decoder_path):
            raise FileNotFoundError(f"No '{os.path.basename(vae_decoder_path)}' file found at {bmodel_path}.")

        #### w4bf16 for 1 device, bf16 for MULTI_DEVICES_NUM devices 
        if quant_dtype == "w4bf16":
            transformer_path = os.path.join(full_model_path, chip_type, f"{flux_type}_{quant_dtype}_transformer.bmodel") 
            if not os.path.isfile(transformer_path):
                raise FileNotFoundError(f"No '{os.path.basename(transformer_path)}' file found at {bmodel_path}.")
        elif quant_dtype == "bf16":
            transformer_paths = [os.path.join(full_model_path, chip_type, f"{flux_type}_{quant_dtype}_transformer_on_device{id}.bmodel") for id in range(self.MULTI_DEVICES_NUM)]
            for single_transformer_path in transformer_paths:            
                if not os.path.isfile(single_transformer_path):
                    raise FileNotFoundError(f"No '{single_transformer_path}' file found at {bmodel_path}.")

        # 3. create handles 
        self.device_ids = device_ids if isinstance(device_ids, list) else [device_ids]
        self.handles = {dev_id: sail.Handle(dev_id) for dev_id in self.device_ids}

        # 4. load normal models
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer_2 = T5TokenizerFast.from_pretrained(tokenizer_2_path)
        #### image_rotary_emd is image_rotary_emb torch.Tensor, shape [1, 4608, 1, 64, 2, 2]
        self.image_rotary_emb = torch.load(rotary_emb_path, map_location = self.EXECUTION_DEVICE)
        self.scheduler = FlowMatchEulerDiscreteScheduler(**self.SCHEDULE_CONFIG[flux_type]) 

        # 5. load clip tokenizer, t5 tokenizer and vae_decoder bmodels
        self.text_encoder =  sail.EngineLLM(clip_path, [self.device_ids[0]]) if quant_dtype == "w4bf16" else sail.EngineLLM(clip_path, [self.device_ids[self.MULTI_DEVICE_ALLOCATION['clip']]])
        self.text_encoder_2 = sail.EngineLLM(t5_path, [self.device_ids[0]]) if quant_dtype == "w4bf16" else sail.EngineLLM(t5_path, [self.device_ids[self.MULTI_DEVICE_ALLOCATION['t5']]])
        self.vae_decoder = sail.EngineLLM(vae_decoder_path, [self.device_ids[0]]) if quant_dtype =="w4bf16" else sail.EngineLLM(vae_decoder_path, [self.device_ids[self.MULTI_DEVICE_ALLOCATION['vae']]])

        # 6. load transformers.
        if quant_dtype == "w4bf16":
            self.transformer_on_device0 = sail.EngineLLM(transformer_path, [self.device_ids[0]])
        else:
            self.transformer_on_device0 = sail.EngineLLM(transformer_paths[0], [self.device_ids[0]])
            self.transformer_on_device1 = sail.EngineLLM(transformer_paths[1], [self.device_ids[1]])
            self.transformer_on_device2 = sail.EngineLLM(transformer_paths[2], [self.device_ids[2]])

        # 7. allocate inputs/outputs device memory, and build input_tensors and output_tensors, the final outputs of clip, t5, vae own sys mem for check precision
        ## 7.1 allocate input/output device mem for clip, output of clip_head, input0 of clip_tail and inputs/outputs of block_${id} are the same, so allocate only once.

        #### token_ids, shape [1, 77]
        clip_head_inputs_map = self.text_encoder.get_input_tensors_addrmode0("clip_head")
        clip_token_ids = sail.Tensor(self.handles[self.device_ids[0 if quant_dtype == "w4bf16" else self.MULTI_DEVICE_ALLOCATION['clip']]], clip_head_inputs_map[0].shape(), clip_head_inputs_map[0].dtype(), False, True)
        
        #### hidden_states, shape [1, 77, 768]
        clip_block_inputs_map = self.text_encoder.get_input_tensors_addrmode0("clip_block_0")
        clip_hidden_states = sail.Tensor(self.handles[self.device_ids[0 if quant_dtype == "w4bf16" else self.MULTI_DEVICE_ALLOCATION['clip']]], clip_block_inputs_map[0].shape(), clip_block_inputs_map[0].dtype(), False, True)

        #### pooling output, shape [1, 768]
        clip_tail_outputs_map = self.text_encoder.get_output_tensors_addrmode0("clip_tail")
        clip_pooling_result = sail.Tensor(self.handles[self.device_ids[0 if quant_dtype == "w4bf16" else self.MULTI_DEVICE_ALLOCATION['clip']]], clip_tail_outputs_map[0].shape(), clip_tail_outputs_map[0].dtype(), True, True)

        ## 7.2 allocate input/output device memory for t5, shape of `inputs/outputs of block_${id}`, `output of t5_head` and `input/output of t5_tail` are the same, so allocate only once.

        #### token_ids, shape [1, 512]
        t5_head_inputs_map = self.text_encoder_2.get_input_tensors_addrmode0("t5_head")
        t5_token_ids = sail.Tensor(self.handles[self.device_ids[0 if quant_dtype == "w4bf16" else self.MULTI_DEVICE_ALLOCATION['t5']]], t5_head_inputs_map[0].shape(), t5_head_inputs_map[0].dtype(), False, True)

        #### hidden_states, shape [1, 512, 4096], same shape with the final output
        t5_block_inputs_map = self.text_encoder_2.get_input_tensors_addrmode0("t5_block_0")
        t5_hidden_states = sail.Tensor(self.handles[self.device_ids[0 if quant_dtype == "w4bf16" else self.MULTI_DEVICE_ALLOCATION['t5']]], t5_block_inputs_map[0].shape(), t5_block_inputs_map[0].dtype(), True, True)

        ## 7.3 allocate input/output device memory for vae
        vae_inputs_map = self.vae_decoder.get_input_tensors_addrmode0("vae_decoder")
        vae_outputs_map = self.vae_decoder.get_output_tensors_addrmode0("vae_decoder")
        latents = sail.Tensor(self.handles[self.device_ids[0 if quant_dtype == "w4bf16" else self.MULTI_DEVICE_ALLOCATION['vae']]], vae_inputs_map[0].shape(), vae_inputs_map[0].dtype(), False, True)
        image = sail.Tensor(self.handles[self.device_ids[0 if quant_dtype == "w4bf16" else self.MULTI_DEVICE_ALLOCATION['vae']]], vae_outputs_map[0].shape(), vae_outputs_map[0].dtype(), True, True)

        ## 7.4 build input and output tensors maps for clip, t5 and vae      
        self.clip_head_inputs = {0: clip_token_ids}
        self.clip_head_outputs = {0: clip_hidden_states}
        self.clip_block_inputs = {0: clip_hidden_states}
        self.clip_block_outputs = {0: clip_hidden_states}
        self.clip_tail_inputs = {0: clip_hidden_states ,1: clip_token_ids}
        self.clip_tail_outputs = {0: clip_pooling_result}

        self.t5_head_inputs = {0: t5_token_ids}
        self.t5_head_outputs = {0: t5_hidden_states}
        self.t5_block_inputs = {0: t5_hidden_states}
        self.t5_block_outputs = {0: t5_hidden_states}
        self.t5_tail_inputs = {0: t5_hidden_states}
        self.t5_tail_outputs = {0: t5_hidden_states}

        self.vae_inputs = {0: latents}
        self.vae_outputs = {0: image}

        ## 7.5 allocate input/output device memory for MM-DiT and single-DiT, allocate input/output device memory for DiT block only once.
        if quant_dtype == "w4bf16":
            ### transformer head
            head_inputs_map = self.transformer_on_device0.get_input_tensors_addrmode0(f"{flux_type}_head")
            #### init latents, shape [1, 4096, 64]
            init_hidden_states = sail.Tensor(self.handles[self.device_ids[0]], head_inputs_map[0].shape(), head_inputs_map[0].dtype(), False, True)
            #### timestep, shape [1]
            timestep = sail.Tensor(self.handles[self.device_ids[0]], head_inputs_map[1].shape(), head_inputs_map[1].dtype(), False, True)
            if flux_type == "dev":
                # guidance scale, shape [1]
                guidance = sail.Tensor(self.handles[self.device_ids[0]], head_inputs_map[2].shape(), head_inputs_map[2].dtype(), False, True)
                #### the pooling output of clip, shape [1, 768]
                pooled_projections = sail.Tensor(self.handles[self.device_ids[0]], head_inputs_map[3].shape(), head_inputs_map[3].dtype(), False, True)
                #### the output of t5, shape [1, 512, 4096]
                init_encoder_hidden_states = sail.Tensor(self.handles[self.device_ids[0]], head_inputs_map[4].shape(), head_inputs_map[4].dtype(), False, True)
            elif flux_type == "schnell":
                #### the pooling output of clip, shape [1, 768]
                pooled_projections = sail.Tensor(self.handles[self.device_ids[0]], head_inputs_map[2].shape(), head_inputs_map[2].dtype(), False, True)
                #### the output of t5, shape [1, 512, 4096]
                init_encoder_hidden_states = sail.Tensor(self.handles[self.device_ids[0]], head_inputs_map[3].shape(), head_inputs_map[3].dtype(), False, True)

            ### image rotary embbeding ,fixed data, read from ids_emb.pt file. shape IMAGE_ROTARY_EMB_SHAPE
            image_rotary_emb = sail.Tensor(self.handles[self.device_ids[0]], self.IMAGE_ROTARY_EMB_SHAPE, sail.Dtype.BM_FLOAT32, False, True)

            ### MM-DiT
            trans_block_on_dev0_inputs_map = self.transformer_on_device0.get_input_tensors_addrmode0(f"{flux_type}_trans_block_0")
            #### hidden_states, shape [1, 4096, 3072]
            hidden_states = sail.Tensor(self.handles[self.device_ids[0]], trans_block_on_dev0_inputs_map[0].shape(), trans_block_on_dev0_inputs_map[0].dtype(), False, True)
            #### encoder_hidden_states, shape [1, 512, 3072]
            encoder_hidden_states = sail.Tensor(self.handles[self.device_ids[0]], trans_block_on_dev0_inputs_map[1].shape(), trans_block_on_dev0_inputs_map[1].dtype(), False, True)
            #### temb shape [1, 3072] 
            temb = sail.Tensor(self.handles[self.device_ids[0]], trans_block_on_dev0_inputs_map[2].shape(), trans_block_on_dev0_inputs_map[2].dtype(), False, True)

            ### single-DiT
            single_trans_block_on_dev0_outputs_map = self.transformer_on_device0.get_output_tensors_addrmode0(f"{flux_type}_single_trans_block_0") 
            #### single-DiT hidden_states, shape [1, 4608, 3072]
            single_DiT_hidden_states = sail.Tensor(self.handles[self.device_ids[0]], single_trans_block_on_dev0_outputs_map[0].shape(), single_trans_block_on_dev0_outputs_map[0].dtype(), False, True)

            ### transformer tail
            tail_outputs_map = self.transformer_on_device0.get_output_tensors_addrmode0(f"{flux_type}_tail")
            #### predict noise in this step, shape [1, 4096, 64]
            predicted_noise = sail.Tensor(self.handles[self.device_ids[0]], tail_outputs_map[0].shape(), tail_outputs_map[0].dtype(), True, True)

            #### build input/output tensors maps
            if flux_type == "dev":
                self.transformer_head_inputs = {0: init_hidden_states, 1: timestep, 2: guidance, 3: pooled_projections, 4: init_encoder_hidden_states}
            elif flux_type == "schnell":
                self.transformer_head_inputs = {0: init_hidden_states, 1: timestep, 2: pooled_projections, 3: init_encoder_hidden_states}
            self.transformer_head_outputs = {0: temb, 1: encoder_hidden_states, 2: hidden_states}
            self.transformer_block_inputs = {0: hidden_states, 1: encoder_hidden_states, 2: temb, 3: image_rotary_emb}
            self.transformer_block_outputs = {0: encoder_hidden_states, 1: hidden_states}
            self.single_transformer_block_inputs = {0: single_DiT_hidden_states, 1: temb, 2: image_rotary_emb}
            self.single_transformer_block_outputs = {0: single_DiT_hidden_states}
            self.transformer_tail_inputs= {0: hidden_states, 1: temb}
            self.transformer_tail_outputs= {0: predicted_noise}
        else:
            ## Assume the head is on chip 0 and the tail is on chip 2, trans_blocks are on chip 0 and 1, single_trans_blocks are on chip 1 and 2.
            ### transformer head
            head_inputs_map = self.transformer_on_device0.get_input_tensors_addrmode0(f"{flux_type}_head")
            #### init latents, shape [1, 4096, 64]
            init_hidden_states = sail.Tensor(self.handles[self.device_ids[0]], head_inputs_map[0].shape(), head_inputs_map[0].dtype(), False, True)
            #### timestep, shape [1]
            timestep = sail.Tensor(self.handles[self.device_ids[0]], head_inputs_map[1].shape(), head_inputs_map[1].dtype(), False, True)
            if flux_type == "dev":
                ####guidance scale, shape [1]
                guidance = sail.Tensor(self.handles[self.device_ids[0]], head_inputs_map[2].shape(), head_inputs_map[2].dtype(), False, True)
                #### the pooling output of clip, shape [1, 768]
                pooled_projections = sail.Tensor(self.handles[self.device_ids[0]], head_inputs_map[3].shape(), head_inputs_map[3].dtype(), False, True)
                #### the output of t5, shape [1, 512, 4096]
                init_encoder_hidden_states = sail.Tensor(self.handles[self.device_ids[0]], head_inputs_map[4].shape(), head_inputs_map[4].dtype(), False, True)
            elif flux_type == 'schnell':
                #### the pooling output of clip, shape [1, 768]
                pooled_projections = sail.Tensor(self.handles[self.device_ids[0]], head_inputs_map[2].shape(), head_inputs_map[2].dtype(), False, True)
                #### the output of t5, shape [1, 512, 4096]
                init_encoder_hidden_states = sail.Tensor(self.handles[self.device_ids[0]], head_inputs_map[3].shape(), head_inputs_map[3].dtype(), False, True)

            ### image rotary embbeding ,fixed data, read from ids_emb.pt file. shape IMAGE_ROTARY_EMB_SHAPE, allocate on all devices
            image_rotary_embbedings_dict = {}
            for idx in range(self.MULTI_DEVICES_NUM):
                image_rotary_embbedings_dict[idx] = sail.Tensor(self.handles[self.device_ids[idx]], self.IMAGE_ROTARY_EMB_SHAPE, sail.Dtype.BM_FLOAT32, False, True)

            ### MM-DiT, on device 0 and 1
            trans_block_on_dev0_inputs_map = self.transformer_on_device0.get_input_tensors_addrmode0(f"{flux_type}_trans_block_0")
            #### hidden_states, shape [1, 4096, 3072], on device0 it needs sys mem, because it would be transfered to device1
            hidden_states_on_dev0 = sail.Tensor(self.handles[self.device_ids[0]], trans_block_on_dev0_inputs_map[0].shape(), trans_block_on_dev0_inputs_map[0].dtype(), True, True)
            hidden_states_on_dev1 = sail.Tensor(self.handles[self.device_ids[1]], trans_block_on_dev0_inputs_map[0].shape(), trans_block_on_dev0_inputs_map[0].dtype(), False, True)
            #### encoder_hidden_states, shape [1, 512, 3072]
            encoder_hidden_states_on_dev0 = sail.Tensor(self.handles[self.device_ids[0]], trans_block_on_dev0_inputs_map[1].shape(), trans_block_on_dev0_inputs_map[1].dtype(), True, True)
            encoder_hidden_states_on_dev1 = sail.Tensor(self.handles[self.device_ids[1]], trans_block_on_dev0_inputs_map[1].shape(), trans_block_on_dev0_inputs_map[1].dtype(), False, True)
            #### temb, allocate on all devices, shape [1, 3072]
            tembs_dict = {}
            for idx in range(self.MULTI_DEVICES_NUM):
                tembs_dict[idx] = sail.Tensor(self.handles[self.device_ids[idx]], trans_block_on_dev0_inputs_map[2].shape(), trans_block_on_dev0_inputs_map[2].dtype(), True if idx == 0 else False, True)

            ### single-DiT, on device 1 and 2
            single_trans_block_on_dev1_outputs_map = self.transformer_on_device1.get_output_tensors_addrmode0(f"{flux_type}_single_trans_block_0") 
            #### single-DiT hidden_states, shape [1, 4608, 3072], 4608 = 4096 + 512
            single_DiT_hidden_states_on_dev1 = sail.Tensor(self.handles[self.device_ids[1]], single_trans_block_on_dev1_outputs_map[0].shape(), single_trans_block_on_dev1_outputs_map[0].dtype(), True, True)
            single_DiT_hidden_states_on_dev2 = sail.Tensor(self.handles[self.device_ids[2]], single_trans_block_on_dev1_outputs_map[0].shape(), single_trans_block_on_dev1_outputs_map[0].dtype(), False, True)

            ### transformer tail, on device 2
            tail_inputs_map = self.transformer_on_device2.get_input_tensors_addrmode0(f"{flux_type}_tail")
            hidden_states_on_dev2 = sail.Tensor(self.handles[self.device_ids[2]], tail_inputs_map[0].shape(), tail_inputs_map[0].dtype(), False, True)
            tail_outputs_map = self.transformer_on_device2.get_output_tensors_addrmode0(f"{flux_type}_tail")
            #### predict noise in this step, shape [1, 4096, 64]
            predicted_noise = sail.Tensor(self.handles[self.device_ids[2]], tail_outputs_map[0].shape(), tail_outputs_map[0].dtype(), True, True)

            #### build input/output tensors maps
            if flux_type == "dev":
                self.transformer_head_inputs = {0: init_hidden_states, 1: timestep, 2: guidance, 3: pooled_projections, 4: init_encoder_hidden_states}
            elif flux_type == "schnell":
                self.transformer_head_inputs = {0: init_hidden_states, 1: timestep, 2: pooled_projections, 3: init_encoder_hidden_states}
            self.transformer_head_outputs = {0: tembs_dict[0], 1: encoder_hidden_states_on_dev0, 2: hidden_states_on_dev0}
            self.transformer_block_inputs_on_dev0 = {0: hidden_states_on_dev0, 1: encoder_hidden_states_on_dev0, 2: tembs_dict[0], 3: image_rotary_embbedings_dict[0]}
            self.transformer_block_outputs_on_dev0 = {0: encoder_hidden_states_on_dev0, 1: hidden_states_on_dev0}
            self.transformer_block_inputs_on_dev1 = {0: hidden_states_on_dev1, 1: encoder_hidden_states_on_dev1, 2: tembs_dict[1], 3: image_rotary_embbedings_dict[1]}
            self.transformer_block_outputs_on_dev1 = {0: encoder_hidden_states_on_dev1, 1: hidden_states_on_dev1}
            self.single_transformer_block_inputs_on_dev1 = {0: single_DiT_hidden_states_on_dev1, 1: tembs_dict[1], 2: image_rotary_embbedings_dict[1]}
            self.single_transformer_block_outputs_on_dev1 = {0: single_DiT_hidden_states_on_dev1}
            self.single_transformer_block_inputs_on_dev2 = {0: single_DiT_hidden_states_on_dev2, 1: tembs_dict[2], 2: image_rotary_embbedings_dict[2]}
            self.single_transformer_block_outputs_on_dev2 = {0: single_DiT_hidden_states_on_dev2}
            self.transformer_tail_inputs= {0: hidden_states_on_dev2, 1: tembs_dict[2]}
            self.transformer_tail_outputs= {0: predicted_noise}

    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
    ):
        r"""
        Get clip_pooling result of prompt

        Args:
            prompt: description of the wanted image, only support str(one prompt) now.
            num_images_per_prompt: only support one now.
            device: which device(where clip is loaded) would receive the output of tokenizer, currently not in effect.

        Returns:
            torch.Tensor: output of clip_pooling, shape is [1, 768].
        """
        # device = device or self.EXECUTION_DEVICE

        # 1. get tokens of prompt
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
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

        # 2. process of clip bmodel, the input token needs converting to float32
        text_input_ids = text_input_ids.numpy().astype(np.float32)
        self.clip_head_inputs[0].update_data(text_input_ids)
        self.text_encoder.process("clip_head", self.clip_head_inputs, self.clip_head_outputs)
        for idx in range(self.CLIP_LAYER_NUM):
            self.text_encoder.process(f"clip_block_{idx}", self.clip_block_inputs, self.clip_block_outputs)
        self.text_encoder.process("clip_tail", self.clip_tail_inputs, self.clip_tail_outputs)

        # 3. convert to torch.Tensor
        self.clip_tail_outputs[0].sync_d2s()
        prompt_embeds = self.clip_tail_outputs[0].asnumpy()
        prompt_embeds = torch.from_numpy(prompt_embeds)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
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
            torch.Tensor: output of t5, shape is [1, 512, 4096].
        """
        # device = device or self.EXECUTION_DEVICE
        dtype = dtype or torch.float32

        # 1. get tokens of prompt
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        
        untruncated_ids = self.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_2.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        # 2. process of t5
        text_input_ids = text_input_ids.numpy().astype(np.float32)
        self.t5_head_inputs[0].update_data(text_input_ids)
        self.text_encoder_2.process("t5_head", self.t5_head_inputs, self.t5_head_outputs)
        for idx in range(self.T5_LAYER_NUM):
            self.text_encoder_2.process(f"t5_block_{idx}", self.t5_block_inputs, self.t5_block_outputs)
        self.text_encoder_2.process("t5_tail", self.t5_tail_inputs, self.t5_tail_outputs) 
        self.t5_tail_outputs[0].sync_d2s()
        prompt_embeds = self.t5_tail_outputs[0].asnumpy()
        prompt_embeds = torch.from_numpy(prompt_embeds)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 512,
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

            # only use the pooled prompt output from the CLIPTextModel
            pooled_prompt_embeds = self._get_clip_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
            )
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

        #### latent_image_ids and text_ids are fixed data, they are processed by pos_embed network, the result of pos_embed network would directly be read from ids_emb.pt file
        # text_ids = torch.zeros(batch_size, prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)
        # text_ids = text_ids.repeat(num_images_per_prompt, 1, 1)

        return prompt_embeds, pooled_prompt_embeds

    def check_inputs(
        self,
        prompt,
        prompt_2,
        height,
        width,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        # Check all inputs.
        if height != 1024 or width != 1024:
            raise ValueError(f"`height` and `width` have to be 1024 currently.")

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
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )

        if max_sequence_length is not None and max_sequence_length > 512:
            raise ValueError(f"`max_sequence_length` cannot be greater than 512 but is {max_sequence_length}")

    @staticmethod
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height // 2, width // 2, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1, 1)
        latent_image_ids = latent_image_ids.reshape(
            batch_size, latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)

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
        height = 2 * (int(height) // self.vae_scale_factor)
        width = 2 * (int(width) // self.vae_scale_factor)

        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            latent_image_ids = self._prepare_latent_image_ids(batch_size, height, width, device, dtype)
            return latents.to(device=device, dtype=dtype), latent_image_ids

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)

        #### latent_image_ids and text_ids are fixed data, they are processed by pos_embed network, the result of pos_embed network would directly be read from ids_emb.pt file
        # latent_image_ids = self._prepare_latent_image_ids(batch_size, height, width, device, dtype)

        return latents

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        height = height // vae_scale_factor
        width = width // vae_scale_factor

        latents = latents.view(batch_size, height, width, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height * 2, width * 2)

        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 3.5,
        seed: int = 42,
        timesteps: List[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = False,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
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
            height: The height in pixels of the generated image, only support 1024 now. 
            width: The width in pixels of the generated image. This is set to 1024 by default for the best results.
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
            max_sequence_length: Maximum sequence length to use with the `prompt`, support 512 now.

        Returns:
            PIL.Image.Image: generated image.
        """
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
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

        prompt_embeds, pooled_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer_config_in_channels // 4
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

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 6. Denoising loop
        ## 6.1 prepare image_rotary_emb data for device
        if self.quant_dtype == "w4bf16":
            self.transformer_block_inputs[3].update_data(self.image_rotary_emb)
        else:
            self.transformer_block_inputs_on_dev0[3].update_data(self.image_rotary_emb)
            self.transformer_block_inputs_on_dev1[3].update_data(self.image_rotary_emb)
            self.single_transformer_block_inputs_on_dev2[2].update_data(self.image_rotary_emb)

        ## 6.2 denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                # handle guidance
                if self.transformer_config_guidance_embeds and self.flux_type != "schnell":
                    guidance = torch.tensor([guidance_scale * 1000], device=device)
                    guidance = guidance.expand(latents.shape[0])
                else:
                    guidance = None

                if self.quant_dtype == "bf16":
                    ## process on device 0                     
                    #### transformer head
                    self.transformer_head_inputs[0].update_data(latents.numpy().astype(np.float32))
                    self.transformer_head_inputs[1].update_data(timestep.numpy().astype(np.float32))
                    if i == 0 and self.flux_type == "dev":
                        self.transformer_head_inputs[2].update_data(guidance.numpy().astype(np.float32)) 
                        self.transformer_head_inputs[3].update_data(pooled_prompt_embeds.numpy().astype(np.float32))
                        self.transformer_head_inputs[4].update_data(prompt_embeds.numpy().astype(np.float32))
                    elif i == 0 and self.flux_type == "schnell":
                        self.transformer_head_inputs[2].update_data(pooled_prompt_embeds.numpy().astype(np.float32))
                        self.transformer_head_inputs[3].update_data(prompt_embeds.numpy().astype(np.float32))

                    self.transformer_on_device0.process(f"{self.flux_type}_head", self.transformer_head_inputs, self.transformer_head_outputs)
                    self.transformer_head_outputs[0].sync_d2s()
                    temb = self.transformer_head_outputs[0].asnumpy()

                    #### MM-dit block
                    for idx in range(self.MM_TRANS_END_FLAG_ON_DEV0 + 1):
                        self.transformer_on_device0.process(f"{self.flux_type}_trans_block_{idx}", self.transformer_block_inputs_on_dev0, self.transformer_block_outputs_on_dev0)

                    self.transformer_block_outputs_on_dev0[0].sync_d2s()
                    encoder_hidden_states = self.transformer_block_outputs_on_dev0[0].asnumpy()
                    self.transformer_block_outputs_on_dev0[1].sync_d2s()
                    hidden_states = self.transformer_block_outputs_on_dev0[1].asnumpy()

                    ## process on device 1
                    #### MM-dit block
                    self.transformer_block_inputs_on_dev1[0].update_data(hidden_states)
                    self.transformer_block_inputs_on_dev1[1].update_data(encoder_hidden_states)
                    self.transformer_block_inputs_on_dev1[2].update_data(temb)

                    for idx in range(self.MM_TRANS_END_FLAG_ON_DEV0 + 1, self.MM_TRANS_LAYERS):
                        self.transformer_on_device1.process(f"{self.flux_type}_trans_block_{idx}", self.transformer_block_inputs_on_dev1, self.transformer_block_outputs_on_dev1)

                    encoder_hidden_states_len = reduce(operator.mul, self.transformer_block_outputs_on_dev1[0].shape(), 1)
                    hidden_states_len = reduce(operator.mul, self.transformer_block_outputs_on_dev1[1].shape(), 1)

                    #### single dit block
                    self.single_transformer_block_inputs_on_dev1[0].sync_d2d(self.transformer_block_outputs_on_dev1[0], 0, 0, encoder_hidden_states_len)
                    self.single_transformer_block_inputs_on_dev1[0].sync_d2d(self.transformer_block_outputs_on_dev1[1], 0, encoder_hidden_states_len, hidden_states_len)
                    self.single_transformer_block_inputs_on_dev1[1].update_data(temb)

                    for idx in range(self.SINGLE_TRANS_END_FLAG_ON_DEV1 + 1):
                        self.transformer_on_device1.process(f"{self.flux_type}_single_trans_block_{idx}", self.single_transformer_block_inputs_on_dev1, self.single_transformer_block_outputs_on_dev1)

                    self.single_transformer_block_outputs_on_dev1[0].sync_d2s()
                    single_trans_hidden_states = self.single_transformer_block_outputs_on_dev1[0].asnumpy()

                    ## process on device 2
                    #### single dit block
                    self.single_transformer_block_inputs_on_dev2[0].update_data(single_trans_hidden_states)
                    self.single_transformer_block_inputs_on_dev2[1].update_data(temb)

                    for idx in range(self.SINGLE_TRANS_END_FLAG_ON_DEV1 + 1, self.SINGLE_TRANS_LAYERS):
                        self.transformer_on_device2.process(f"{self.flux_type}_single_trans_block_{idx}", self.single_transformer_block_inputs_on_dev2, self.single_transformer_block_outputs_on_dev2)

                    #### transformer tail
                    self.transformer_tail_inputs[0].sync_d2d(self.single_transformer_block_outputs_on_dev2[0], encoder_hidden_states_len, 0, hidden_states_len)
                    self.transformer_on_device2.process(f"{self.flux_type}_tail", self.transformer_tail_inputs, self.transformer_tail_outputs)
                    self.transformer_tail_outputs[0].sync_d2s()
                    noise_pred = self.transformer_tail_outputs[0].asnumpy()
                    noise_pred = torch.from_numpy(noise_pred)

                elif self.quant_dtype == "w4bf16":
                    self.transformer_head_inputs[0].update_data(latents.numpy().astype(np.float32))
                    self.transformer_head_inputs[1].update_data(timestep.numpy().astype(np.float32))
                    if i == 0 and self.flux_type == "dev":
                        self.transformer_head_inputs[2].update_data(guidance.numpy().astype(np.float32))
                        self.transformer_head_inputs[3].update_data(pooled_prompt_embeds.numpy().astype(np.float32))
                        self.transformer_head_inputs[4].update_data(prompt_embeds.numpy().astype(np.float32))
                    elif i == 0 and self.flux_type == "schnell":
                        self.transformer_head_inputs[2].update_data(pooled_prompt_embeds.numpy().astype(np.float32))
                        self.transformer_head_inputs[3].update_data(prompt_embeds.numpy().astype(np.float32))
                    self.transformer_on_device0.process(f"{self.flux_type}_head", self.transformer_head_inputs, self.transformer_head_outputs)

                    for idx in range(self.MM_TRANS_LAYERS):
                        self.transformer_on_device0.process(f"{self.flux_type}_trans_block_{idx}", self.transformer_block_inputs, self.transformer_block_outputs)
                    
                    encoder_hidden_states_len = reduce(operator.mul, self.transformer_block_outputs[0].shape(), 1)
                    hidden_states_len = reduce(operator.mul, self.transformer_block_outputs[1].shape(), 1)

                    self.single_transformer_block_inputs[0].sync_d2d(self.transformer_block_outputs[0], 0, 0, encoder_hidden_states_len)
                    self.single_transformer_block_inputs[0].sync_d2d(self.transformer_block_outputs[1], 0, encoder_hidden_states_len, hidden_states_len)

                    for idx in range(self.SINGLE_TRANS_LAYERS):
                        self.transformer_on_device0.process(f"{self.flux_type}_single_trans_block_{idx}", self.single_transformer_block_inputs, self.single_transformer_block_outputs)
                    
                    self.transformer_tail_inputs[0].sync_d2d(self.single_transformer_block_outputs[0], encoder_hidden_states_len, 0, hidden_states_len)
                    self.transformer_on_device0.process(f"{self.flux_type}_tail", self.transformer_tail_inputs, self.transformer_tail_outputs)
                    self.transformer_tail_outputs[0].sync_d2s()
                    noise_pred = self.transformer_tail_outputs[0].asnumpy()
                    noise_pred = torch.from_numpy(noise_pred)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if output_type == "latent":
            image = latents

        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            #### combined to vae bmodel
            # latents = (latents / self.vae_config_scaling_factor) + self.vae_config_shift_factor
            #### process of vae
            self.vae_inputs[0].update_data(latents.numpy().astype(np.float32))
            self.vae_decoder.process("vae_decoder", self.vae_inputs, self.vae_outputs)
            self.vae_outputs[0].sync_d2s()
            image = torch.from_numpy(self.vae_outputs[0].asnumpy())
            image = self.image_processor.postprocess(image, output_type=output_type)

        # if not return_dict:
        #     return (image,)

        return image