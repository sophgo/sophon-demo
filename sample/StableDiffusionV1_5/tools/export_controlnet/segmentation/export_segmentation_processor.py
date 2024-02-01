import os
import torch
from transformers import UperNetForSemanticSegmentation

save_dir = "processors"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

device = "cpu"
data_type = torch.float32

model= UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small", torch_dtype = data_type)
model = model.to(device)

model.eval()
for para in model.parameters():
    para.requires_grad = False

torch.onnx.export(model, torch.randn(1,3,576,576).to(data_type).to(device), f"{save_dir}/segmentation_processor.onnx", input_names=['input_image'], output_names = ['seg_result'], opset_version=11)