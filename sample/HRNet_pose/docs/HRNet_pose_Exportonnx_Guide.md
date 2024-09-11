# HRNet_pose模型onnx导出
HRNet_pose的onnx模型导出是在Pytorch的环境下进行的，需根据[deep-high-resolution-net.pytorch](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)的要求安装好Pytorch环境，准备好相应的代码和模型，并保证模型能够在Pytorch环境下正常推理运行。也可参考第三方代码库[deep-learning-for-image-processing](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_keypoint/HRNet)导出onnx模型

示例：

'''

import torch
import torch.onnx

from model import HighResolutionNet  
model = HighResolutionNet()  # 导入模型，注意根据实际情况修改

model.load_state_dict(torch.load("./models/pytorch/pose_coco/pose_hrnet_w32_256x192.pth"))

model.eval()

dummy_input = torch.randn(1, 3, 256, 192)

torch.onnx.export(model, dummy_input, "pose_hrnet_w32_256x192.onnx", opset_version=11, verbose=True)

'''
