#===----------------------------------------------------------------------===#
#
# Copyright (C) 2024 Sophgo Technologies Inc.  All rights reserved.
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

import gradio as gr

from flux_pipeline import FluxPipeline

def load_pipeline(args):
    pipeline = FluxPipeline()
    pipeline.from_models(
        full_model_path = args.model_path,
        chip_type = args.chip_type,
        device_ids = args.dev_ids, 
        quant_dtype = args.quant_type,
        flux_type = args.flux_type,
        use_tiny_vae = args.tiny_vae
    )
    return pipeline

def infer(prompt, num_steps, guidance, seed):
    seed = int(seed)
    seed = random.randint(0, 2**32 - 1) if seed == -1 else seed
    image = pipeline(prompt, prompt, int(num_steps), float(guidance), seed)[0]
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"generated_image_{current_time}.png"
    image.save(filename)
    return image, filename

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
    # dev_ids
    parser.add_argument("--dev_ids", type=int, nargs='+', default=[0,1,2], help="device ids, support one or three devices, such as 0 or 1 2 3")
    # use tiny vae
    parser.add_argument("--tiny_vae", type=bool, default=False, help="use taef1 model if it is True, please set to be True in soc mode")
    try:
        args = parser.parse_args()
    except SystemExit as e:
        # This exception will be raised if --help or invalid command line arguments
        # are used. Currently streamlit prevents the program from exiting normally
        # so we have to do a hard exit.
        os._exit(e.code)

    log.basicConfig(level=log.INFO)

    pipeline = load_pipeline(args)

    with gr.Blocks() as demo:
        gr.Markdown(f"# Flux Image Generation Demo - Model: flux.1-{args.flux_type}")
        
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", value="a photo of a forest with mist swirling around the tree trunks. The word \"FLUX\" is painted over it in big, red brush strokes with visible texture")
                
                with gr.Accordion("Advanced Options", open=True):
                    num_steps = gr.Slider(1, 50, 4 if args.flux_type == "schnell" else 10, step=1, label="Number of steps")
                    guidance = gr.Slider(1.0, 10.0, 3.5, step=0.1, label="Guidance", interactive=not args.flux_type=="schnell")
                    seed = gr.Textbox(-1, label="Seed (-1 for random)")
                
                generate_btn = gr.Button("Generate")
            
            with gr.Column():
                output_image = gr.Image(label="Generated Image")
                download_btn = gr.File(label="Download full-resolution")

        generate_btn.click(
            fn=infer,
            inputs=[prompt, num_steps, guidance, seed],
            outputs=[output_image, download_btn],
        )
    demo.launch(server_name="0.0.0.0")