from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
import torch

device = "cpu"
dtype = torch.float32

# if torch.cuda.is_available() is True and memory is enough.
# device = "cuda:0"
# dtype = torch.float16

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=dtype)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=dtype)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to(device)

def build_controlnet_input():
    control_model_input = torch.randn(2, 4, 64, 64)
    t = torch.tensor([99])
    controlnet_prompt_embeds = torch.randn(2, 77, 768)
    image = torch.randn(2, 3, 512, 512)
    return control_model_input, t, controlnet_prompt_embeds, image

def convert_tensor_into_torch(args):
    return [arg.to(device).to(dtype) for arg in args]

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
            return res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7], res[8], res[9], res[10], res[11], mid

    traced_model = torch.jit.trace(build_controlnet, (control_model_input,t,controlnet_prompt_embeds,image))
    traced_model.save("openpose_controlnet.pt")

export_controlnet()