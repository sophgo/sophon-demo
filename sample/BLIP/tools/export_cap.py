#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
#!/bin/python3
from models.blip import blip_decoder

from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import numpy as np

device = torch.device('cpu')

image_size = 384
raw_image = Image.open("demo.jpg").convert('RGB')
transform = transforms.Compose([
    transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])
image = transform(raw_image).unsqueeze(0).to(device)

model_url = 'checkpoints/model_base_14M.pth'
#model_url = 'checkpoints/model_base.pth'
#model_url = 'checkpoints/model_large_caption.pth'

model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')

for params in model.parameters():
    params.requires_grad = False
model.eval()
model = model.to(device)

prompt = 'a picture of '
class TdecWrapper(torch.nn.Module):
    def __init__(self, model):
        super(TdecWrapper, self).__init__()

    def forward(self, pixel_values):
        input_ids = model.tokenizer(prompt, return_tensors="pt").input_ids
        input_ids[:,0] = model.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1]
        image_embeds = model.visual_encoder(pixel_values) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long)
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask":image_atts}
        outputs = model.text_decoder.generate(input_ids=input_ids,
                                                  max_length=20,
                                                  min_length=5,
                                                  num_return_sequences=1,
                                                  eos_token_id=model.tokenizer.sep_token_id,
                                                  pad_token_id=model.tokenizer.pad_token_id,
                                                  repetition_penalty=1.0,
                                                  **model_kwargs)
        return outputs

tdecmodel = TdecWrapper(model)
tdecmodel.eval()
tdecmodel = tdecmodel.to(device)


pixel_values=image
with torch.no_grad():
    outputs = tdecmodel(pixel_values)

print(outputs)
captions = []
for output in outputs[0]:
    caption = model.tokenizer.decode(output, skip_special_tokens=True)
    captions.append(caption)
print(captions)

with torch.no_grad():
    torch.onnx.export(tdecmodel,
            pixel_values,
            "blip_cap.onnx",
            opset_version=13,
            do_constant_folding=True,
            export_params=True,
            input_names=["pixel_values"],
            output_names=["output"],
            dynamic_axes={"pixel_values":{0:"batch_size"},
                "output":{0:"batch_size"}})
