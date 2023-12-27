import torch

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)

device = "cpu"
dtype = torch.float32

# if torch.cuda.is_available() is True and memory is enough.
# device = "cuda:0"
# dtype = torch.float16

checkpoint = "lllyasviel/control_v11p_sd15_scribble"

controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

pipe.to(device)

def build_controlnet_input():
    control_model_input = torch.randn(2, 4, 64, 64)
    t = torch.tensor([99])
    controlnet_prompt_embeds = torch.randn(2, 77, 768)
    image = torch.randn(2, 3, 512, 512)
    return control_model_input, t, controlnet_prompt_embeds, image

def export_controlnet():

    controlnet = pipe.controlnet
    controlnet = controlnet.eval()
    for para in controlnet.parameters():
        para.requires_grad = False
    control_model_input,t,controlnet_prompt_embeds,image = convert_tensor_into_torch(build_controlnet_input())

    def build_controlnet(latent,t,prompt_embeds,image):
        with torch.no_grad():
            res, mid = controlnet(latent,
                             t,
                             encoder_hidden_states=prompt_embeds,
                             controlnet_cond=image,
                             conditioning_scale=1,
                             guess_mode=None,
                             return_dict=False)
            # import pdb;pdb.set_trace()
            return res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7], res[8], res[9], res[10], res[11], mid

    traced_model = torch.jit.trace(build_controlnet, (control_model_input,t,controlnet_prompt_embeds,image))
    traced_model.save("scribble_controlnet.pt")

def convert_tensor_into_torch(args):
    return [arg.to(device).to(dtype) for arg in args]

export_controlnet()