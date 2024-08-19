# YOL0v8模型导出
## 1. 准备工作
TPU-MLIR版本 ≥ 1.10，还需要安装Pytorch和numpy库
TPU-MLIR 最新release whl下载链接：https://github.com/sophgo/tpu-mlir/releases
```bash
pip install torch numpy
```
## 2. 导出onnx模型
为了使得mask运算能够使用tpu进行加速，需要构建必要的运算操作，并且导出为onnx模型。
```python
import torch
import torch.nn.functional as F
import numpy as np

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mask_info, output1):
        '''
            mask_info 1 * box_size * 32
            output1   1 * 32 * 160(feature_h) * 160(feature_w)
            m_confThreshold   1 * 1
        '''
        ptotos = output1[0].view(32, -1)
 
        feature = torch.matmul(mask_info, ptotos)
        feature_uint8= torch.clamp(feature.mul(255).to(torch.uint8),min=0,max=255)
    
        return feature_uint8

mask_info=torch.rand(1,1,32)
output1=torch.rand(1,32,160,160)

ret = {"mask_info":mask_info,"output1":output1}
model = Model()

torch.onnx.export(
    model,    
    (mask_info,output1),
    "yolov8_int8_getmask_32.onnx",
    verbose=True, 
    input_names=["mask_info", "output1"], 
    output_names=["output"], 
    opset_version=11,
    dynamic_axes={
        "mask_info": { 1:"num"},
        "output": {1:"num"}
    }
)
```

上述脚本会在在当前目录下生成yolov8_int8_getmask_32.onnx。

在上述export代码在进行矩阵乘后，又进行了数乘、截断，使得所有的数值量化到0~255之间，这样能保证输出的数值量化精度的稳定性。

INT8输出的模型，可以直接使用bmcv的函数接口直接进行处理，可以大大加快速度

## 3. 导出bmodel
详见scripts/gen_maskbmodel.sh

