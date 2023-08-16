# BERT4torch模型onnx导出
torch==1.13.0
'''
import torch
import torchvision
 
model = torch.jit.load('bert4torch_jit.pt') # load model
input_data = torch.zeros([1, 256]) # generate input data
output_onnx_path = 'test.onnx'
 
torch.onnx.export(model, input_data, output_onnx_path, verbose=True, opset_version=12)
'''
