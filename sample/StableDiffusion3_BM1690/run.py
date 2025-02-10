#===----------------------------------------------------------------------===#
#
# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import argparse
import atexit
import logging as log
import os
import random
import signal
import sys
import time

import numpy as np
import torch

from stable_diffusion3_pipeline import StableDiffusion3Pipeline

pipeline = None

def cleanup():
    global pipeline
    try:
        if pipeline is not None:
            pipeline.__del__()
            pipeline = None
    except NameError:
        print("pipeline is not defined or has been cleaned.")

def signal_handler(sig, frame):
    cleanup()
    sys.exit()

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
    parser.add_argument("--guidance_scale", type=float, default=7.0, help="guidance for each step, 2 batch when it greater than 1")
    # dev_ids
    parser.add_argument("--dev_ids", type=int, nargs='+', default= [0, 1], help="TPU ID, support 1 or 2 devices, such as 0 or 0,1")
    # fix seed
    parser.add_argument("--seed", type=int, default=42, help="seed value, must be between 0 and 2^32 - 1")
    try:
        args = parser.parse_args()
    except SystemExit as e:
        # This exception will be raised if --help or invalid command line arguments
        # are used. Currently streamlit prevents the program from exiting normally
        # so we have to do a hard exit.
        os._exit(e.code)

    log.basicConfig(level=log.INFO)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    atexit.register(cleanup)
    
    if len(args.dev_ids) == 2:
        import multiprocessing
        multiprocessing.set_start_method('spawn')

    try:
        pipeline = load_pipeline(args)

        for idx in range(1):
            pipeline_start = time.time()
            result = run(pipeline, args)
            pipeline_time = time.time() - pipeline_start
            print(f"The {idx}th:")
            log.info("pipeline time(s): {:.2f}".format(pipeline_time))
            result[0].save(f'result_{idx}.png')

    except Exception as e:
        print(f"main process error: {e}")

    finally:
        cleanup()
