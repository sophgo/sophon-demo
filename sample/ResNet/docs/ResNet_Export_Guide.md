# ResNet模型导出
## 1. 准备工作
ResNet模型导出是在Pytorch模型的生产环境下进行的，需提前安装好Pytorch环境。 
## 2. 主要步骤 
### 2.1 导出torchscript模型
​Pytorch模型在编译前要经过`torch.jit.trace`，trace后的模型才能使用tpu-nntc编译BModel。trace的方法和原理可参考[torch.jit.trace参考文档](../../../docs/torch.jit.trace_Guide.md)。
### 2.2 导出onnx模型
如果使用tpu-mlir编译模型，则必须先将Pytorch模型导出为onnx模型。下面以导出1 batch的onnx模型为例进行演示：
```bash
import torch
import torch.onnx
from torchvision.models import resnet50

if __name__ == '__main__':
    input = torch.randn(1, 3, 224, 224)          # [1,3,224,224]分别对应[B,C,H,W]
    model = resnet50()                           # 载入模型框架
    model.load_state_dict(torch.load("xxx.pth")) # xxx.pth表示.pth文件, 这一步载入模型权重
    model.eval()                                 # 设置模型为推理模式
    torch.onnx.export(model, input, "xxx.onnx")  # xxx.onnx表示.onnx文件, 这一步导出为onnx模型
```
