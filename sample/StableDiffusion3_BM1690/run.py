#===----------------------------------------------------------------------===#
#
# Copyright (C) 2024 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import argparse
import logging as log
import os
import random
import time

import numpy as np
import torch
from stable_diffusion3_pipeline import StableDiffusion3Pipeline

def load_pipeline(args):
    pipeline = StableDiffusion3Pipeline()
    load_start = time.time()
    pipeline.from_models(
        full_model_path = args.model_path,
        chip_type = args.chip_type,
        device_ids = args.dev_ids, 
    )
    load_time = time.time() - load_start
    log.info("load model time(s): {:.2f}".format(load_time))
    return pipeline

def run(pipeline, args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    result = pipeline(
        prompt = args.prompt,
        #### set prompt_2/3 to be None if they are empty
        prompt_2 = None if args.prompt_2 == "" else args.prompt_2,
        prompt_3 = None if args.prompt_3 == "" else args.prompt_3,
        negative_prompt = args.negative_prompt,
        #### set negative_prompt_2/3 to be None if they are empty
        negative_prompt_2 = None if args.negative_prompt_2 == "" else args.negative_prompt_2,
        negative_prompt_3 = None if args.negative_prompt_3 == "" else args.negative_prompt_3,
        num_inference_steps = args.num_inference_steps,
        guidance_scale = args.guidance_scale,
    )[0]
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model_path
    parser.add_argument("--model_path", type=str, default="./models", help="bmodels path")
    # chip_type 
    parser.add_argument("--chip_type", type=str, default="BM1690", help="product type")
    # positive prompt for clip_l
    parser.add_argument("--prompt", type=str, default="A cute ant holding a sign that says Sophgo", help="prompt for clip_l")
    # positive prompt for clip_g
    parser.add_argument("--prompt_2", type=str, default=None, help="prompt for clip_g")
    # positive prompt for t5
    parser.add_argument("--prompt_3", type=str, default=None, help="prompt for t5")
    # negative prompt for clip_l
    parser.add_argument("--negative_prompt", type=str, default="worst quality", help="negative prompt for clip_l")
    # negative prompt for clip_g
    parser.add_argument("--negative_prompt_2", type=str, default=None, help="negative prompt for clip_g")
    # negative prompt for t5
    parser.add_argument("--negative_prompt_3", type=str, default=None, help="negative prompt for t5")
    # num_inference_steps
    parser.add_argument("--num_inference_steps", type=int, default=20, help="total denoising steps")
    # guidance_scale
    parser.add_argument("--guidance_scale", type=float, default=7.0, help="guidance for each step, must >= 1")
    # dev_ids
    parser.add_argument("--dev_ids", type=int, nargs='+', default= 0, help="TPU ID")
    # fix seed
    parser.add_argument("--seed", type=int, default=42, help="seed value, must be between 0 and 2**32 - 1")
    try:
        args = parser.parse_args()
    except SystemExit as e:
        # This exception will be raised if --help or invalid command line arguments
        # are used. Currently streamlit prevents the program from exiting normally
        # so we have to do a hard exit.
        os._exit(e.code)

    log.basicConfig(level=log.INFO)

    pipe = load_pipeline(args)

    pipe_start = time.time()
    result = run(pipe, args)
    pipe_time = time.time() - pipe_start
    log.info("pipeline time(s): {:.2f}".format(pipe_time))

    result[0].save('result.png')
