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
import time

from flux_pipeline import FluxPipeline

def load_pipeline(args):
    pipeline = FluxPipeline()
    load_start = time.time()
    pipeline.from_models(
        full_model_path = args.model_path,
        chip_type = args.chip_type,
        device_ids = args.dev_ids, 
        quant_dtype = args.quant_type,
        flux_type = args.flux_type,
        use_tiny_vae = args.tiny_vae,
    )
    load_time = time.time() - load_start
    log.info("load model time(s): {:.2f}".format(load_time))
    return pipeline

def run(pipeline, args):
    result = pipeline(
        prompt = args.prompt,
        prompt_2 = None if args.prompt_2 == "" else args.prompt_2,
        num_inference_steps = args.num_inference_steps,
        guidance_scale = args.guidance_scale,
        seed = args.seed,
    )[0]
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # flux_type
    parser.add_argument("--flux_type", type=str, default="dev", help="dev or schnell")
    # model_path
    parser.add_argument("--model_path", type=str, default="../models", help="bmodels path")
    # chip_type 
    parser.add_argument("--chip_type", type=str, default="BM1684X", help="product type")
    # quant_type 
    parser.add_argument("--quant_type", type=str, default="bf16", help="bf16 or w4bf16, transformer module")
    # tokenizer
    parser.add_argument("--tokenizer", type=str, default="../models/tokenizer", help="tokenizer")
    # tokenizer_2
    parser.add_argument("--tokenizer_2", type=str, default="../models/tokenizer_2", help="tokenizer_2")
    # prompt
    parser.add_argument("--prompt", type=str, default="a powerful mysterious sorceress, casting lightning magic, detailed clothing, digital painting, hyperrealistic, fantasy, Surrealist, upper body, artstation, highly detailed, sharp focus, stunningly beautiful, dystopian", help="prompt for clip")
    # prompt_2
    parser.add_argument("--prompt_2", type=str, default="", help="prompt_2 for t5, would be the same with prompt if not set")
    # num_inference_steps
    parser.add_argument("--num_inference_steps", type=int, default=10, help="total denoising steps")
    # guidance_scale
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="guidance for each step")
    # dev_id
    parser.add_argument("--dev_ids", type=int, nargs='+', default=[0,1,2], help="device ids, support one or three devices, such as 0 or 1 2 3")
    # use tiny vae
    parser.add_argument("--tiny_vae", type=bool, default=False, help="use taef1 model if it is True, please set to be True in soc mode")
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

    result.save('result.png')