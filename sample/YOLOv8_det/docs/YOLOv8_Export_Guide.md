# YOL0v8模型导出
## 1. 准备工作
YOL0v8模型导出是在Pytorch模型的生产环境下进行的，需提前根据[​YOLOv8官方开源仓库](https://github.com/ultralytics/ultralytics)的要求安装好Pytorch环境，准备好相应的代码和模型，并保证模型能够在Pytorch环境下正常推理运行。

同时需要安装官方提供的第三方库：
```bash
# 该第三方库需要 Python >= 3.7, PyTorch >= 1.7
pip3 install ultralytics
```

## 2. 导出onnx模型
如果使用tpu-mlir编译模型，则必须先将Pytorch模型导出为onnx模型。YOL0v8官方仓库提供了模型导出接口，可以直接使用它导出onnx模型：

```python
from ultralytics import YOLO
# Load a model
model = YOLO("yolov8s.pt")  # 载入预训练模型
# Export the model
success = model.export(format="onnx", batch=1)  # 导出静态ONNX模型，需设置batch参数
```

上述脚本会在原始pt模型所在目录下生成导出的onnx模型，导出后可以修改模型名称以区分不同参数和输出类型，如`yolov8s_1b.onnx`表示单输出且输入batch大小为1的onnx模型。

## 3. 常见问题
TODO
