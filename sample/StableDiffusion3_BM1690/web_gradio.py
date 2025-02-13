#===----------------------------------------------------------------------===#
#
# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import argparse
import datetime
import logging as log
import os
import random
import signal
import time

import gradio as gr
import numpy as np
import torch
from stable_diffusion3_pipeline import StableDiffusion3Pipeline

def signal_handler(sig, frame):
    demo.close()
    exit(0)

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

def run(prompt, negative_prompt, num_inference_steps, guidance_scale, seed):
    seed = int(seed)
    seed = random.randint(0, 2**32 - 1) if seed < 0 else seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    pipeline_start_time = time.time()
    result = pipeline(
        prompt = prompt,
        negative_prompt = negative_prompt,
        num_inference_steps = num_inference_steps,
        guidance_scale = guidance_scale,
    )[0]
    pipeline_time = time.time() - pipeline_start_time
    log.info("inference time(s): {:.2f}".format(pipeline_time))

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"generated_image_{current_time}.png"
    result[0].save(filename)
    return result[0], filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model_path
    parser.add_argument("--model_path", type=str, default="./models", help="bmodels path")
    # chip_type 
    parser.add_argument("--chip_type", type=str, default="BM1690", help="product type")
    # dev_ids
    parser.add_argument("--dev_ids", type=int, nargs='+', default= [0, 1], help="TPU ID, support 1 or 2 devices, such as 0 or 0,1")
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

    pipeline = load_pipeline(args)

    with gr.Blocks() as demo:
        gr.Markdown(f"# SD3.0 Image Generation Demo - Model: stable-diffusion-3-medium")

        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", value="A cat holding a sign that says hello world")
                negative_prompt = gr.Textbox(label="Negative prompt", value="worst quality")

                with gr.Accordion("Advanced Options", open=True):
                    num_steps = gr.Slider(1, 50, 20, step=1, label="Number of steps")
                    guidance = gr.Slider(0.1, 10.0, 7.0, step=0.1, label="Guidance")
                    seed = gr.Textbox(value = -1, label="Seed (-1 for random), between [0, 4294967295]")

                generate_btn = gr.Button("Generate")

            with gr.Column():
                output_image = gr.Image(label="Generated Image")
                download_btn = gr.File(label="Download full-resolution")

        generate_btn.click(
            fn=run,
            inputs=[prompt, negative_prompt, num_steps, guidance, seed],
            outputs=[output_image, download_btn],
        )
    demo.launch(server_name="0.0.0.0")
