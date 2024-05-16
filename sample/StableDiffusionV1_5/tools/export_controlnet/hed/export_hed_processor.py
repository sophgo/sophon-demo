import torch
from controlnet_aux import HEDdetector
import numpy as np
import os

save_dir = "processors"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

img_size = [(128, 384), (128, 448), (128, 512), (192, 384), (192, 448), (192, 512), (256, 384), 
            (256, 448), (256, 512), (320, 384), (320, 448), (320, 512), (384, 384), (384, 448), 
            (384, 512), (448, 448), (448, 512), (512, 512), (512, 576), (512, 640), (512, 704),
            (512, 768), (512, 832), (512, 896), (768, 768), (384, 128), (448, 128), (512, 128),
            (384, 192), (448, 192), (512, 192), (384, 256), (448, 256), (512, 256), (384, 320),
            (448, 320), (512, 320), (448, 384), (512, 384), (512, 448), (576, 512), (640, 512), 
            (704, 512), (768, 512), (832, 512), (896, 512)]

hed = HEDdetector.from_pretrained('lllyasviel/Annotators')

def export_hed_processor():
    hed_processor = hed.netNetwork

    hed_processor.eval()
    for parameter in hed_processor.parameters():
        parameter.requires_grad=False

    for img_height, img_width in img_size:
        img_height = float(img_height)
        img_width = float(img_width)
        k = float(512) / min(img_height, img_width)
        img_height *= k
        img_width *= k
        img_height = int(np.round(img_height / 64.0)) * 64
        img_width = int(np.round(img_width / 64.0)) * 64

        input = torch.randn(1, 3, img_height, img_width)

        def build_hed_flow(input):
            with torch.no_grad():
                out =hed_processor(input)[0]
            return out

        traced_model=torch.jit.trace(build_hed_flow, (input))

        traced_model.save(f"./{save_dir}/hed_processor_{img_height}_{img_width}.pt")

export_hed_processor()