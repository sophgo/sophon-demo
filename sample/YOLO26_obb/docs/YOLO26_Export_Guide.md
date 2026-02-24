# YOLO26模型导出
## 1. 准备工作
YOLO26模型导出是在Pytorch模型的生产环境下进行的，需提前根据[​YOLO26官方开源仓库](https://github.com/ultralytics/ultralytics)的要求安装好环境，准备好相应的代码和模型，并保证模型能够在Pytorch环境下正常推理运行。本例程导出环境版本为：`torch-2.9.1+cu128, onnx 1.18.0`。


## 2. 导出onnx模型
如果使用tpu-mlir编译模型，则必须先将Pytorch模型导出为onnx模型。YOLO26官方仓库提供了模型导出接口，可以直接使用它导出onnx模型，应设置opset为13。

```python
from ultralytics import YOLO
# Load a model
model = YOLO("yolo26s-obb.pt")
# Export the model
model.export(format="onnx",opset=13)  # 注意
```