import torch
from transformers import UperNetForSemanticSegmentation

device = "cpu"
dtype = torch.float32

import pdb
pdb.set_trace()

model= UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")
model = model.to(device)

model.eval()
for para in model.parameters():
    para.requires_grad = False

torch.onnx.export(model, torch.randn(1,3,576,576).to(dtype).to(device), "seg_processor.onnx", input_names=['input_image'], output_names = ['seg_result'])