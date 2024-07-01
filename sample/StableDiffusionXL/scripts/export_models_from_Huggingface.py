import os
from diffusers import StableDiffusionXLPipeline
import torch

unet_model_save_path = "../models/onnx_pt/unet"
vae_encoder_model_save_path = "../models/onnx_pt/vae_encoder"
vae_decoder_model_save_path = "../models/onnx_pt/vae_decoder"
text_encoder_1_save_path = "../models/onnx_pt/text_encoder_1"
text_encoder_2_save_path = "../models/onnx_pt/text_encoder_2"

if not os.path.exists(unet_model_save_path):
    os.makedirs(unet_model_save_path)
if not os.path.exists(vae_encoder_model_save_path):
    os.makedirs(vae_encoder_model_save_path)
if not os.path.exists(vae_decoder_model_save_path):
    os.makedirs(vae_decoder_model_save_path)
if not os.path.exists(text_encoder_1_save_path):
    os.makedirs(text_encoder_1_save_path)
if not os.path.exists(text_encoder_2_save_path):
    os.makedirs(text_encoder_2_save_path)

device = 'cpu'
data_type = torch.float32

input_shapes = {
    "text_encoder_1":[(1,77)],
    "text_encoder_2":[(1,77)],
    "base":[(2,4,128,128),(1),(2,77,2048),(2,1280),(2,6)],
    "refiner":[(2,4,128,128),(1),(2,77,1280),(2,1280),(2,5)],
    "vae_decoder":[(1,4,128,128)],
    "vae_encoder":[(1,3,1024,1024)],
}

def eval_mode(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

def build_input(input_shape):
    fake_input = []
    for shape in input_shape:
        fake_input.append(torch.randn(shape,dtype = data_type, device = device))
    return fake_input

def export_base_unet(model, input_shape = input_shapes['base']):
    eval_mode(model)
    dummy_input = build_input(input_shape)
    def build_base_unet(
            latent_model_input,
            t,
            prompt_embeds,
            text_embeds,
            time_ids,
        ):
        with torch.no_grad():
            result = model(
                            latent_model_input,
                            t,
                            prompt_embeds,
                            added_cond_kwargs = {
                                "text_embeds":text_embeds,
                                "time_ids":time_ids
                            },
                        )[0]
        return result
    traced_model = torch.jit.trace(build_base_unet, dummy_input)
    traced_model.save(os.path.join(unet_model_save_path, "unet_base.pt"))

def export_vae_encoder(model, input_shape = input_shapes['vae_encoder']):
    eval_mode(model)
    dummy_input = build_input(input_shape)
    def build_vae_encoder(input):
        with torch.no_grad():
            x = model.encoder(input)
            return model.quant_conv(x)
    traced_model = torch.jit.trace(build_vae_encoder, dummy_input)
    traced_model.save(os.path.join(vae_encoder_model_save_path, "vae_encoder.pt"))

def export_vae_decoder(model, input_shape = input_shapes['vae_decoder']):
    eval_mode(model)
    dummy_input = build_input(input_shape)
    def build_vae_decoder(input):
        with torch.no_grad():
            x = model.post_quant_conv(input)
            return model.decoder(x)
    traced_model = torch.jit.trace(build_vae_decoder, dummy_input)
    traced_model.save(os.path.join(vae_decoder_model_save_path, "vae_decoder.pt"))

def export_text_encoder_1(model, input_shape = input_shapes['text_encoder_1']):
    eval_mode(model)
    dummy_input = torch.randint(0, 1000, input_shape[0])
    def build_text_encoder_1(input):
        with torch.no_grad():
            prompt_embeds = model(input,output_hidden_states = True)
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            return pooled_prompt_embeds, prompt_embeds
    traced_model = torch.jit.trace(build_text_encoder_1, dummy_input)
    torch.onnx.export(traced_model, dummy_input, os.path.join(text_encoder_1_save_path, 'text_encoder_1.onnx'))

def export_text_encoder_2(model, input_shape = input_shapes['text_encoder_2']):
    eval_mode(model)
    dummy_input = torch.randint(0, 1000, input_shape[0])
    def build_text_encoder_2(input):
        with torch.no_grad():
            result = model(input,output_hidden_states = True)
            pooled_prompt_embeds = result[0]
            prompt_embeds = result.hidden_states[-2]
            return pooled_prompt_embeds, prompt_embeds
    traced_model = torch.jit.trace(build_text_encoder_2, dummy_input)
    torch.onnx.export(traced_model, dummy_input, os.path.join(text_encoder_2_save_path, 'text_encoder_2.onnx'))

base = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", load_safety_checker=False, torch_dtype = torch.float32, use_safetensors = True)

export_base_unet(base.unet)
export_vae_decoder(base.vae)
export_vae_encoder(base.vae)
export_text_encoder_1(base.text_encoder)
export_text_encoder_2(base.text_encoder_2)
