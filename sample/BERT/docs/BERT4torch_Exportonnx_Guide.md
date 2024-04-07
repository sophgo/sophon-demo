# bert4torch模型onnx导出
bert4torch模型导出是在Pytorch模型的生产环境下进行的，需提前根据[A simple training framework that recreates bert4keras in PyTorch. bert4torch](https://github.com/Tongjilibo/bert4torch/)的要求安装好Pytorch环境，准备好相应的代码和模型，并保证模型能够在Pytorch环境下正常推理运行。
torch==1.13.0
'''
import torch
import torchvision
 
model = torch.jit.load('bert4torch_jit.pt') # load model
input_data = torch.zeros([1, 256]) # generate input data
output_onnx_path = 'test.onnx'
 
torch.onnx.export(model, input_data, output_onnx_path, verbose=True, opset_version=12)
'''
