from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, PNDMScheduler
import torch
import os

save_dir = "controlnets"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

img_size = [(128, 384), (128, 448), (128, 512), (192, 384), (192, 448), (192, 512), (256, 384), 
            (256, 448), (256, 512), (320, 384), (320, 448), (320, 512), (384, 384), (384, 448), 
            (384, 512), (448, 448), (448, 512), (512, 512), (512, 576), (512, 640), (512, 704),
            (512, 768), (512, 832), (512, 896), (768, 768), (384, 128), (448, 128), (512, 128),
            (384, 192), (448, 192), (512, 192), (384, 256), (448, 256), (512, 256), (384, 320),
            (448, 320), (512, 320), (448, 384), (512, 384), (512, 448), (576, 512), (640, 512), 
            (704, 512), (768, 512), (832, 512), (896, 512)]

# cuda is alternative for device if cuda is available
device = "cpu"
# torch float16 is alternative for data_type if cuda is available
data_type = torch.float32

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_scribble", torch_dtype=data_type
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=data_type
)

pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
pipe.to(device)

def build_controlnet_input(img_height, img_width):
    control_model_input = torch.randn(2, 4, img_height//8, img_width//8)
    t = torch.tensor([99])
    controlnet_prompt_embeds = torch.randn(2, 77, 768)
    image = torch.randn(2, 3, img_height, img_width)
    return control_model_input, t, controlnet_prompt_embeds, image

def convert_tensor_into_torch(args):
    return [arg.to(device).to(data_type) for arg in args]

def export_controlnet():
    controlnet = pipe.controlnet
    controlnet = controlnet.eval()
    for para in controlnet.parameters():
        para.requires_grad = False

    for img_height, img_width in img_size:
        control_model_input, t, controlnet_prompt_embeds, image = convert_tensor_into_torch(
            build_controlnet_input(img_height, img_width))

        def build_controlnet(latent, t, prompt_embeds, image):
            with torch.no_grad():
                res, mid = controlnet(latent,
                                    t,
                                    encoder_hidden_states=prompt_embeds,
                                    controlnet_cond=image,
                                    conditioning_scale=1,
                                    guess_mode=None,
                                    return_dict=False)
                return res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7], res[8], res[9], res[10], res[11], mid

        traced_model = torch.jit.trace(
            build_controlnet, (control_model_input, t, controlnet_prompt_embeds, image))
        traced_model.save(f"./{save_dir}/scribble_controlnet_{img_height}_{img_width}.pt")

export_controlnet()