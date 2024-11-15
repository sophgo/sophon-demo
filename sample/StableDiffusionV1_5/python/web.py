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
import signal
import time
import threading

import gradio as gr
import numpy as np
import torch
from web_stable_diffusion import StableDiffusionPipeline
from diffusers import PNDMScheduler,EulerDiscreteScheduler
from argparse import Namespace
timer = None

need_load=True
args=Namespace(controlnet_img=None, controlnet_name=None, dev_id=0, guidance_scale=0.0, model_path='../models/BM1688/',img_size=(512, 512), init_img=None, neg_prompt='None', num_inference_steps=1, processor_name=None, prompt='a cat', sd_turbo=1, stage='singlize', strength=1.0, tokenizer='../models/tokenizer_path',vae_decoder_model_path='../models/BM1688/singlize/vae_decoder_1688_f16.bmodel',vae_encoder_model_path='../models/BM1688/singlize/vae_encoder_1688_f16.bmodel',unet_model_path='../models/BM1688/singlize/unet_1688_core_1_f16.bmodel',text_encoder_model_path='../models/BM1688/singlize/text_encoder_1688_bf16.bmodel')
engine=None
provided_img_size = [(128, 384), (128, 448), (128, 512), (192, 384), (192, 448), (192, 512), (256, 384), 
            (256, 448), (256, 512), (320, 384), (320, 448), (320, 512), (384, 384), (384, 448), 
            (384, 512), (448, 448), (448, 512), (512, 512), (512, 576), (512, 640), (512, 704),
            (512, 768), (512, 832), (512, 896), (768, 768), (384, 128), (448, 128), (512, 128),
            (384, 192), (448, 192), (512, 192), (384, 256), (448, 256), (512, 256), (384, 320),
            (448, 320), (512, 320), (448, 384), (512, 384), (512, 448), (576, 512), (640, 512), 
            (704, 512), (768, 512), (832, 512), (896, 512)]
def timeout_callback():
    global need_load,engine
    print(need_load)
    if(need_load == False):
        need_load=True
        del engine
      
    print("5秒内没有收到新的请求")
def reset_timer():
    global timer
    # 如果计时器已经存在，先取消它
    if timer:
        timer.cancel()
    # 重新设置计时器
    timer = threading.Timer(10, timeout_callback)
    timer.start()
def run(prompt, negative_prompt, num_inference_steps, guidance_scale, seed,width,height,strength,vae_decoder_model_path,vae_encoder_model_path,unet_model_path,text_encoder_model_path,sample):
    global need_load,engine,args
    print(need_load)
    
    args.prompt=prompt
    args.neg_prompt=negative_prompt
    args.num_inference_steps=num_inference_steps
    args.guidance_scale=guidance_scale
    args.seed=seed
    args.strength=strength
    if(sample=="sd_turbo"):
        args.sd_turbo=1
    else :
        args.sd_turbo=0

    args.vae_decoder_model_path=vae_decoder_model_path
    args.vae_encoder_model_path=vae_encoder_model_path
    args.text_encoder_model_path=text_encoder_model_path

    args.unet_model_path=unet_model_path
    if need_load:
        engine = load_pipeline(args)
     
   
    print(negative_prompt)
    if args.img_size:
        height, width = args.img_size
        if (height, width) not in provided_img_size:
            print(f'{height},{width} is not supported.')
    else:
        print('Please provide image size using --img_size.')
    if args.prompt:
        np.random.seed(int(seed))
        image = engine(
            prompt = args.prompt,
            height = height,
            width = width,
            negative_prompt = args.neg_prompt,
            init_image = args.init_img,
            controlnet_img = args.controlnet_img,
            strength = args.strength,
            num_inference_steps = args.num_inference_steps,
            guidance_scale = args.guidance_scale
        )
    image.save("result.png")
    reset_timer()

    return image,"result.png"
def signal_handler(sig, frame):
    demo.close()
    exit(0)

def load_pipeline(args):
    global need_load
    print("开始初始化...")

    scheduler = PNDMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        skip_prk_steps = True,
    )
    if(args.sd_turbo):
        scheduler = EulerDiscreteScheduler(#sd-turbo
            beta_end=0.012,
            beta_start=0.00085,
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",
            final_sigmas_type="zero",
            interpolation_type="linear",
            prediction_type="epsilon",
            rescale_betas_zero_snr=False,
            steps_offset=1,
            timestep_spacing="trailing",
            timestep_type="discrete"
        )
    pipeline = StableDiffusionPipeline(
        scheduler = scheduler,
        model_path = args.model_path,
        vae_decoder_path = args.vae_decoder_model_path,
        vae_encoder_path = args.vae_encoder_model_path,
        unet_path = args.unet_model_path,
        text_encoder_path = args.text_encoder_model_path,
        stage = args.stage,
        controlnet_name = args.controlnet_name,
        processor_name = args.processor_name,
        dev_id = args.dev_id,
        tokenizer = args.tokenizer,
        seed=args.seed
    )
    need_load=False
    print(need_load)
    print("初始化完成")
    return pipeline



if __name__ == "__main__":
  

    log.basicConfig(level=log.INFO)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # engine = load_pipeline(args)
    # reset_timer()

    with gr.Blocks() as demo:
        gr.Markdown(f"# SD3.0 Image Generation Demo - Model: stable-diffusion-3-medium")

        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", value="A cat holding a sign that says hello world")
                negative_prompt = gr.Textbox(label="Negative prompt", value="worst quality")

                with gr.Accordion("Advanced Options", open=True):
                    num_steps = gr.Slider(1, 50, 1, step=1, label="Number of steps")
                    guidance = gr.Slider(0, 10.0, 0.0, step=0.1, label="Guidance")
                    strength = gr.Slider(0, 10.0, 1.0, step=0.1, label="Strength")
                    width = gr.Slider(1, 1024, 512, step=1, label="Width")
                    height = gr.Slider(1, 1024, 512, step=1, label="Height")
                    seed = gr.Textbox(value = 2048, label="Seed (0 for random), between [0, 4294967295]")
                    vae_decoder_model_path = gr.Textbox(value = "../models/BM1688/singlize/vae_decoder_1688_f16.bmodel", label="Vae_decoder_model_path")
                    vae_encoder_model_path = gr.Textbox(value = "../models/BM1688/singlize/vae_encoder_1688_f16.bmodel", label="Vae_encoder_model_path")

                    unet_model_path = gr.Textbox(value = "../models/BM1688/singlize/unet_1688_core_1_f16.bmodel", label="Unet_model_path")
                    text_decoder_model_path = gr.Textbox(value = "../models/BM1688/singlize/text_encoder_1688_bf16.bmodel", label=" text_decoder_model_path")

                    sample = gr.Textbox(value = "sd_turbo", label="Sample model")

                generate_btn = gr.Button("Generate")

            with gr.Column():
                output_image = gr.Image(label="Generated Image")
                download_btn = gr.File(label="Download full-resolution")

        generate_btn.click(
            fn=run,
            inputs=[prompt, negative_prompt, num_steps, guidance, seed,width,height,strength,vae_decoder_model_path,vae_encoder_model_path,unet_model_path,text_decoder_model_path,sample],
            outputs=[output_image, download_btn],
        )
    demo.launch(server_name="0.0.0.0")