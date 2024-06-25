# YOL0v8模型导出
## 1. 准备工作
需要安装Pytorch和numpy库
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

    def forward(self, x0, x1):
        mask_info = x0
        ptotos = x1[0].view(32, -1)
        x = torch.matmul(mask_info, ptotos)
        return x


output1=torch.rand(1,32,160,160)
mask_info=torch.rand(1,10,32)
model = Model()


torch.onnx.export(
    model,    
    (mask_info,output1),
    "yolov8_getmask.onnx",
    verbose=True, 
    input_names=["mask_info", "output1"], 
    output_names=["output"], 
    opset_version=11,
    dynamic_axes={
        "mask_info": {0:"batch", 1:"num"},
        "output1": {0:"batch"},
        "output": {0:"batch", 1:"num"}
    }
)
```

上述脚本会在在当前目录下生成yolov8_getmask.onnx。

## 3. 导出bmodel
详见scripts/gen_maskbmodel.sh

