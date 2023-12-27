import torch
import numpy as np
import os
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

model_save_path = "../models/onnx_pt"
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

def build_unet_input():
    latent= np.random.rand(2, 4, 64, 64)
    t     = np.array([99])
    prompt_embeds=np.random.rand(2, 77, 768)
    return latent,t,prompt_embeds

def convert_into_torch(args):
    return [torch.from_numpy(arg).to(torch.float32) for arg in args]

def convert_into_long_torch(args):
    return [torch.from_numpy(arg).to(torch.long) for arg in args]

def export_unet():

    unet = pipe.unet
    unet = unet.eval()
    for para in unet.parameters():
        para.requires_grad = False
    latent,t,prompt_embeds = convert_into_torch(build_unet_input())

    def build_unet(latent,t,prompt_embeds):
        with torch.no_grad():
            res = unet(latent,t,encoder_hidden_states=prompt_embeds,return_dict=False)[0]
            return res
    traced_model = torch.jit.trace(build_unet, (latent,t,prompt_embeds))
    traced_model.save(os.path.join(model_save_path, "unet_fp32.pt"))

def build_encoder_input():
    shapes = [1,77]
    prompt_embeds = np.ones(shapes)
    return prompt_embeds

def export_textencoder():
    for para in pipe.text_encoder.parameters():
        para.requires_grad = False
    batch = 1
    fake_input = torch.randint(0, 1000, (batch, 77))
    onnx_model_path = os.path.join(model_save_path, "./text_encoder_1684x_f32.onnx")
    torch.onnx.export(pipe.text_encoder, fake_input, onnx_model_path, verbose=True, opset_version=14, input_names=["input_ids"], output_names=["output"])

def export_vae_decoder():
    vae = pipe.vae
    vae = vae.to(torch.float32)
    vae = vae.eval()
    for para in vae.parameters():
        para.requires_grad = False
    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder2 = vae.post_quant_conv
            self.decoder1 = vae.decoder

        def forward(self, x):
            x = self.decoder2(x)
            x = self.decoder1(x)
            return x
    vae_decoder_model = Decoder()
    img_size = (512,512)
    fake_input = torch.randn(1, 4, img_size[0] // 8, img_size[1] // 8).to(torch.float32)
    traced_model = torch.jit.trace(vae_decoder_model, (fake_input))
    traced_model.save(os.path.join(model_save_path, "vae_decoder_singlize.pt"))

def export_vae_encoder():
    vae = pipe.vae
    vae = vae.eval()
    for para in vae.parameters():
        para.requires_grad = False
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder1 = vae.encoder
            self.encoder2 = vae.quant_conv

        def forward(self, x):
            x = self.encoder1(x)
            x = self.encoder2(x)
            return x

    encoder = Encoder()
    img_size = (512,512)
    fake_input = torch.randn(1, 3, img_size[0], img_size[1]).to(torch.float32)
    traced_model = torch.jit.trace(encoder, (fake_input))
    traced_model.save(os.path.join(model_save_path, "vae_encoder_singlize.pt"))

#pipe.enable_model_cpu_offload()
# pipe.enable_xformers_memory_efficient_attention()

print("================Start Save UNet Model======================")
export_unet()
print("================UNet has been saved!=======================")

print("================Start Save Text Encoder====================")
export_textencoder()
print("===============Text Encoder has been saved!================")

print("================Start Save Vae Decoder=====================")
export_vae_decoder()
print("================Vae Decocer has been saved=================")

print("================Start Save Vae Encoder=====================")
export_vae_encoder()
print("================Vae Encocer has been saved=================")
