import argparse
import os
import numpy as np
from stable_diffusion import StableDiffusionPipeline
from diffusers import PNDMScheduler

def run(engine, args):
    if args.prompt:
        np.random.seed()
        image = engine(
            prompt = args.prompt,
            height = 512,
            width = 512,
            negative_prompt = args.neg_prompt,
            init_image = args.init_image,
            controlnet_img = args.controlnet_img,
            strength = args.strength,
            num_inference_steps = args.num_inference_steps,
            guidance_scale = args.guidance_scale
        )
    return image

def load_pipeline(args):
    scheduler = PNDMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        skip_prk_steps = True,
    )
    pipeline = StableDiffusionPipeline(
        scheduler = scheduler,
        model_path = args.model_path,
        stage = args.stage,
        controlnet_name = args.controlnet_name,
        processor_name = args.processor_name,
        dev_id = args.dev_id,
        tokenizer = args.tokenizer
    )
    return pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model_path
    parser.add_argument("--model_path", type=str, default="../models/BM1684X", help="bmodels path")
    # unet_name
    parser.add_argument("--stage", type=str, default="singlize", help="singlize / multilize input size")
    # controlnet_name
    parser.add_argument("--controlnet_name", type=str, default=None, help="controlnet name")
    # processor_name
    parser.add_argument("--processor_name", type=str, default=None, help="processor name")
    # controlnet_img
    parser.add_argument("--controlnet_img", type=str, default=None, help="Input image for controlnet")
    # tokenizer
    parser.add_argument("--tokenizer", type=str, default="../models/tokenizer_path", help="tokenizer")
    # prompts
    parser.add_argument("--prompt", type=str, default="a rabbit drinking at the bar", help="prompt for this model")
    # negtive prompts
    parser.add_argument("--neg_prompt", type=str, default="worst quality", help="negative prompt for this model")
    # num_inference_steps
    parser.add_argument("--num_inference_steps", type=int, default=20, help="total denoising steps")
    # strength
    parser.add_argument("--strength", type=float, default=0.7, help="denoise strength")
    # guidance_scale
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="guidance for each step")
    # image_path
    parser.add_argument("--init_image", type=str, default=None, help="image path")
    # dev_id
    parser.add_argument("--dev_id", type=int, default=3, help="device id")
    try:
        args = parser.parse_args()
    except SystemExit as e:
        # This exception will be raised if --help or invalid command line arguments
        # are used. Currently streamlit prevents the program from exiting normally
        # so we have to do a hard exit.
        os._exit(e.code)

    engine = load_pipeline(args)
    image = run(engine, args)
    image.save("result.png")
