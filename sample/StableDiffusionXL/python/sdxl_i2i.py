from StableDiffusionPipelineImg2Img import StableDiffusionXLImg2ImgPipeline
import argparse
import os
from PIL import Image
from transformers import CLIPTokenizer
from diffusers.schedulers import EulerDiscreteScheduler

scheduler_config = {'Euler D':{
        "_class_name": "EulerDiscreteScheduler",
        "_diffusers_version": "0.24.0",
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
        "beta_start": 0.00085,
        "interpolation_type": "linear",
        "num_train_timesteps": 1000,
        "prediction_type": "epsilon",
        "steps_offset": 1,
        "timestep_spacing": "leading",
        "use_karras_sigmas": False
    }}

def load_pipe(args):
    tokenizer_1 = CLIPTokenizer.from_pretrained(args.tokenizer)
    tokenzier_2 = CLIPTokenizer.from_pretrained(args.tokenizer_2)

    pipe = StableDiffusionXLImg2ImgPipeline(
        vae_encoder_path = os.path.join(args.model_path, "vae_encoder_1684x_bf16.bmodel"),
        vae_decoder_path = os.path.join(args.model_path, "vae_decoder_1684x_bf16.bmodel"),
        te_encoder_path = os.path.join(args.model_path, "text_encoder_1_1684x_f32.bmodel"),
        te_encoder_2_path = os.path.join(args.model_path, "text_encoder_2_1684x_f16.bmodel"),
        tokenizer = tokenizer_1,
        tokenizer_2 = tokenzier_2,
        unet_path = os.path.join(args.model_path, "unet_base_1684x_bf16.bmodel"),
        scheduler = EulerDiscreteScheduler(**(scheduler_config["Euler D"])),
        dev_id = args.dev_id)

    return pipe

def run(engine, args):
    init_img = Image.open(args.init_img)
    if args.prompt:
        image = engine(
            prompt = args.prompt,
            negative_prompt = args.neg_prompt,
            image = init_img,
            num_inference_steps = args.num_inference_steps,
            guidance_scale = args.guidance_scale,
            strength = args.strength
        )
    return image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model_path
    parser.add_argument("--model_path", type=str, default="../models/BM1684X", help="bmodels path")
    # tokenizer
    parser.add_argument("--tokenizer", type=str, default="../models/tokenizer", help="tokenizer")
    # tokenizer_2
    parser.add_argument("--tokenizer_2", type=str, default="../models/tokenizer_2", help="tokenizer_2")
    # prompts
    parser.add_argument("--prompt", type=str, default="A magician riding a grey donkey", help="prompt for this model")
    # negtive prompts
    parser.add_argument("--neg_prompt", type=str, default="worst quality", help="negative prompt for this model")
    # init image path
    parser.add_argument("--init_img", type=str, default="../pics/astronaut.png", help="referenced image path")
    # num_inference_steps
    parser.add_argument("--num_inference_steps", type=int, default=50, help="total denoising steps")
    # guidance_scale
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="guidance for each step")
    # strength
    parser.add_argument("--strength", type=float, default=0.7, help="strength for referenced image, it is an inverse proportional weight")
    # dev_id
    parser.add_argument("--dev_id", type=int, default=0, help="device id")
    try:
        args = parser.parse_args()
    except SystemExit as e:
        # This exception will be raised if --help or invalid command line arguments
        # are used. Currently streamlit prevents the program from exiting normally
        # so we have to do a hard exit.
        os._exit(e.code)

pipe = load_pipe(args)
result = run(pipe, args)
result.save("i2i_result.png")