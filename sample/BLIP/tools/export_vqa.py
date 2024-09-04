#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
#!/bin/python3
from models.blip_vqa import blip_vqa

from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import numpy as np

device = torch.device('cpu')

image_size = 480
raw_image = Image.open("demo.jpg").convert('RGB')
transform = transforms.Compose([
    transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])
image = transform(raw_image).unsqueeze(0).to(device)

model_url = 'checkpoints/model_vqa.pth'
#model_url = 'checkpoints/model_base_vqa_capfilt_large.pth'
#model_url = 'checkpoints/blip_okvqa.pth'
#model_url = 'checkpoints/blip_aokvqa.pth'
    
model = blip_vqa(pretrained=model_url, image_size=image_size, vit='base')
for params in model.parameters():
    params.requires_grad = False
model.eval()
model = model.to(device)

#question = 'where is the woman sitting?'
#question = 'What is the dog doing?'
question = 'What is this?'

class TencWrapper(torch.nn.Module):
    def __init__(self, model):
        super(TencWrapper, self).__init__()

    def forward(self, image_embeds, input_ids, attention_mask):
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long)
        question_output = model.text_encoder(input_ids,
                attention_mask = attention_mask,
                encoder_hidden_states = image_embeds,
                encoder_attention_mask = image_atts,
                return_dict = True)
        return question_output.last_hidden_state


class TdecWrapper(torch.nn.Module):
    def __init__(self, model):
        super(TdecWrapper, self).__init__()
        self.model = model

    def forward(self, question_states):
        bos_ids = torch.full((image.size(0),1),fill_value=model.tokenizer.bos_token_id,device=image.device)
        question_atts = torch.ones(question_states.shape[:-1],dtype=torch.long).to(image.device)
        model_kwargs = {"encoder_hidden_states": question_states, "encoder_attention_mask":question_atts}
        outputs = model.text_decoder.generate(input_ids=bos_ids, max_length=10, min_length=1, num_beams=1,
            eos_token_id=model.tokenizer.sep_token_id, pad_token_id=model.tokenizer.pad_token_id, **model_kwargs)
        return outputs

tencmodel = TencWrapper(model)
tencmodel.eval()
tencmodel = tencmodel.to(device)

tdecmodel = TdecWrapper(model)
tdecmodel.eval()
tdecmodel = tdecmodel.to(device)

question = model.tokenizer(question, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(image.device)
question.input_ids[:,0] = model.tokenizer.enc_token_id

pixel_values = image
input_ids = question.input_ids
attention_mask = question.attention_mask

with torch.no_grad():
    image_embeds = model.visual_encoder(pixel_values)

with torch.no_grad():
    question_states = tencmodel(image_embeds, input_ids, attention_mask)

with torch.no_grad():
    outputs = tdecmodel(question_states)

answers = []
for output in outputs[0]:
    answers.append(model.tokenizer.decode(output, skip_special_tokens=True))
print(answers)

with torch.no_grad():
    torch.onnx.export(model.visual_encoder,
            pixel_values,
            "blip_vqa_venc.onnx",
            opset_version=13,
            export_params=True,
            do_constant_folding=True,
            input_names=["pixel_values"],
            output_names=["output"],
            dynamic_axes={"pixel_values":{0: "batch_size"},
                "output":{0:"batch_size"}})

with torch.no_grad():
    torch.onnx.export(tencmodel,
            (image_embeds, input_ids, attention_mask),
            "blip_vqa_tenc.onnx",
            opset_version=13,
            export_params=True,
            do_constant_folding=True,
            input_names=["image_embeds", "input_ids", "attention_mask"],
            output_names=["output"],
            dynamic_axes={"image_embeds":{0: "batch_size"},
                "input_ids":{0: "batch_size"},
                "attention_mask":{0: "batch_size"},
                "output":{0:"batch_size"}})

with torch.no_grad():
    torch.onnx.export(tdecmodel,
            question_states,
            "blip_vqa_tdec.onnx",
            opset_version=13,
            export_params=True,
            do_constant_folding=True,
            input_names=["question_states"],
            output_names=["output"],
            dynamic_axes={"question_states":{0:"batch_size"},
                "output":{0:"batch_size"}})
