import torch
import numpy as np
import os
import torch
from diffusers import StableDiffusionPipeline, PNDMScheduler

unet_model_save_path = "../models/onnx_pt/multilize/unet"
vae_encoder_model_save_path = "../models/onnx_pt/multilize/vae_encoder"
vae_decoder_model_save_path = "../models/onnx_pt/multilize/vae_decoder"

if not os.path.exists(unet_model_save_path):
    os.makedirs(unet_model_save_path)
if not os.path.exists(vae_encoder_model_save_path):
    os.makedirs(vae_encoder_model_save_path)
if not os.path.exists(vae_decoder_model_save_path):
    os.makedirs(vae_decoder_model_save_path)

text_encoder_save_path = "../models/onnx_pt"
if not os.path.exists(text_encoder_save_path):
    os.makedirs(text_encoder_save_path)

model_id = "runwayml/stable-diffusion-v1-5"

# torch float16 is alternative for data_type if cuda is available
data_type = torch.float32

# cuda is alternative for device if cuda is available
device = "cpu"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=data_type)
pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
pipe.to(device)

supported_img_size = [(128, 384), (128, 448), (128, 512), (192, 384), (192, 448), (192, 512), (256, 384),
            (256, 448), (256, 512), (320, 384), (320,
                                                 448), (320, 512), (384, 384), (384, 448),
            (384, 512), (448, 448), (448, 512), (512,
                                                 512), (512, 576), (512, 640), (512, 704),
            (512, 768), (512, 832), (512, 896), (768,
                                                 768), (384, 128), (448, 128), (512, 128),
            (384, 192), (448, 192), (512, 192), (384,
                                                 256), (448, 256), (512, 256), (384, 320),
            (448, 320), (512, 320), (448, 384), (512,
                                                 384), (512, 448), (576, 512), (640, 512),
            (704, 512), (768, 512), (832, 512), (896, 512)]

img_size = [(512, 512), (768, 768)]

def convert_into_torch(args):
    return [torch.from_numpy(arg).to(data_type).to(device) for arg in args]


def build_unet_input(img_height, img_width):
    latent = np.random.rand(2, 4, img_height//8, img_width//8)
    t = np.array([99])
    prompt_embeds = np.random.rand(2, 77, 768)
    res = []
    res.append(np.random.random((2, 320, img_height//8,
                                 img_width//8)).astype(np.float32))
    res.append(np.random.random((2, 320, img_height//8,
                                 img_width//8)).astype(np.float32))
    res.append(np.random.random((2, 320, img_height//8,
                                 img_width//8)).astype(np.float32))
    res.append(np.random.random((2, 320, img_height//16,
                                 img_width//16)).astype(np.float32))
    res.append(np.random.random((2, 640, img_height//16,
                                 img_width//16)).astype(np.float32))
    res.append(np.random.random((2, 640, img_height//16,
                                 img_width//16)).astype(np.float32))
    res.append(np.random.random((2, 640, img_height//32,
                                 img_width//32)).astype(np.float32))
    res.append(np.random.random((2, 1280, img_height//32,
                                 img_width//32)).astype(np.float32))
    res.append(np.random.random((2, 1280, img_height//32,
                                 img_width//32)).astype(np.float32))
    res.append(np.random.random((2, 1280, img_height//64,
                                 img_width//64)).astype(np.float32))
    res.append(np.random.random((2, 1280, img_height//64,
                                 img_width//64)).astype(np.float32))
    res.append(np.random.random((2, 1280, img_height//64,
                                 img_width//64)).astype(np.float32))
    res.append(np.random.random((2, 1280, img_height//64,
                                 img_width//64)).astype(np.float32))
    return latent, t, prompt_embeds, res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7], res[8], res[9], res[10], res[11], res[12]


def export_unet():
    unet = pipe.unet
    unet = unet.eval()
    for para in unet.parameters():
        para.requires_grad = False

    for img_height, img_width in img_size:

        converted_tensor = convert_into_torch(
            build_unet_input(img_height, img_width))

        latent, t, prompt_embeds, res_samples = converted_tensor[
            0], converted_tensor[1], converted_tensor[2], converted_tensor[3:]

        down_block_res_samples = res_samples[:-1]
        mid_block_res_sample = res_samples[-1]

        def build_unet(latent, t, prompt_embeds, mid_block_res_sample, *down_block_res):

            down_block_res_samples = []

            for item in down_block_res:
                down_block_res_samples.append(item)

            with torch.no_grad():
                res = unet(latent, t, encoder_hidden_states=prompt_embeds, down_block_additional_residuals=down_block_res_samples,
                           mid_block_additional_residual=mid_block_res_sample, return_dict=False)[0]
                return res

        traced_model = torch.jit.trace(
            build_unet, (latent, t, prompt_embeds, mid_block_res_sample, *down_block_res_samples))

        traced_model.save(os.path.join(
            unet_model_save_path, f"unet_{img_height}_{img_width}.pt"))


def export_textencoder():
    for para in pipe.text_encoder.parameters():
        para.requires_grad = False
    batch = 1
    fake_input = torch.randint(0, 1000, (batch, 77))
    onnx_model_path = os.path.join(
        text_encoder_save_path, "./text_encoder_1684x_f32.onnx")
    torch.onnx.export(pipe.text_encoder, fake_input, onnx_model_path, verbose=True,
                      opset_version=14, input_names=["input_ids"], output_names=["output"])


def export_vae_decoder():
    vae = pipe.vae
    vae = vae.to(data_type)
    vae = vae.eval()
    for para in vae.parameters():
        para.requires_grad = False

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder2 = vae.post_quant_conv
            self.decoder1 = vae.decoder

        def forward(self, x):
            with torch.no_grad():
                x = self.decoder2(x)
                x = self.decoder1(x)
                return x

    vae_decoder_model = Decoder()
    for img_height, img_width in img_size:

        fake_input = torch.randn(
            1, 4, img_height // 8, img_width // 8).to(data_type).to(device)

        traced_model = torch.jit.trace(vae_decoder_model, (fake_input))

        traced_model.save(os.path.join(
            vae_decoder_model_save_path, f"vae_decoder_{img_height}_{img_width}.pt"))


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
            with torch.no_grad():
                x = self.encoder1(x)
                x = self.encoder2(x)
                return x

    encoder = Encoder()

    for img_height, img_width in img_size:

        fake_input = torch.randn(
            1, 3, img_height, img_width).to(data_type).to(device)

        traced_model = torch.jit.trace(encoder, (fake_input))

        traced_model.save(os.path.join(
            vae_encoder_model_save_path, f"vae_encoder_{img_height}_{img_width}.pt"))


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
