# DirectMHP模型导出
## 1. 准备工作
DirectMHP模型导出是在Pytorch模型的生产环境下进行的，需提前根据[DirectMHP官方开源仓库](https://github.com/hnuzhy/DirectMHP)的要求安装好Pytorch环境，准备好相应的代码和模型，并保证模型能够在Pytorch环境下正常推理运行。本例程导出环境版本为：`torch==1.7.1+cpu`。

## 2. 导出onnx模型
如果使用tpu-mlir编译模型，则必须先将Pytorch模型导出为onnx模型。

```python
import torch
import torch.onnx
import argparse
from utils.torch_utils import select_device
from models.experimental import attempt_load
# 加载PyTorch模型
device = select_device('cpu', batch_size=1)
model = attempt_load('./torch/directmhp_torchscript.pt', map_location=device)
model.eval()
# 创建一个输入张量
dummy_input = torch.randn(1, 3, 1280, 1280)
# 导出ONNX模型
onnx_path = "./directmhp.onnx"
torch.onnx.export(model, dummy_input, onnx_path, verbose=True, opset_version=12,export_params=True)
```

上述脚本会在原始pt模型所在目录下生成导出的onnx模型`directmhp.onnx`。