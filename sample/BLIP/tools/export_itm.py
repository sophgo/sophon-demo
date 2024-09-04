#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
#!/bin/python3
from models.blip_itm import blip_itm

from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

device = torch.device('cpu')

image_size = 384
raw_image = Image.open("demo.jpg").convert('RGB')
transform = transforms.Compose([
    transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])
image = transform(raw_image).unsqueeze(0).to(device)

model_url = 'checkpoints/model_base_retrieval_coco.pth'
#model_url = 'checkpoints/model_base_retrieval_flickr.pth'
    
model = blip_itm(pretrained=model_url, image_size=image_size, vit='base')
for params in model.parameters():
    params.requires_grad = False
model.eval()
model = model.to(device='cpu')

caption = 'a woman sitting on the beach with a dog'
print('text: %s' %caption)

itm_output = model(image,caption,match_head='itm')
itm_score = torch.nn.functional.softmax(itm_output,dim=1)[:,1]
print('The image and text is matched with a probability of %.4f'%itm_score)

'''
itc_score = model(image,caption,match_head='itc')
print('The image feature and text feature has a cosine similarity of %.4f'%itc_score)
'''

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()

    def forward(self, pixel_values, input_ids, attention_mask):
        image_embeds = model.visual_encoder(pixel_values)
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        output = model.text_encoder(input_ids,
                attention_mask = attention_mask,
                encoder_hidden_states = image_embeds,
                encoder_attention_mask = image_atts,
                return_dict = True)
        itm_output = model.itm_head(output.last_hidden_state[:,0,:])
        itm = torch.nn.functional.softmax(itm_output,dim=1)[:,1]
        return itm

newmodel=ModelWrapper(model)
newmodel.eval()

text = model.tokenizer(caption, padding='max_length', truncation=True, max_length=35,return_tensors="pt")

pixel_values = image
input_ids = text.input_ids
attention_mask = text.attention_mask

itm = newmodel(pixel_values, input_ids, attention_mask)
print('The image and text is matched with a probability of %.4f'%itm)

with torch.no_grad():
    torch.onnx.export(newmodel,
            (pixel_values, input_ids, attention_mask),
            "blip_itm.onnx",
            opset_version=13,
            input_names=["pixel_values","input_ids", "attention_mask"],
            output_names=["output"],
            dynamic_axes={"pixel_values":{0:"batch_size"},
                "input_ids":{0:"batch_size"},
                "attention_mask":{0:"batch_size"},
                "output":{0:"batch_size"}})
