# YOL0v8模型导出
## 1. 准备工作
可选择从[YOLOv8官方主页](https://github.com/ultralytics/ultralytics?tab=readme-ov-file)下载YOLOv8-obb.pt模型，或在[导出onnx模型](#2-导出onnx模型)中自动下载模型。
安装如下依赖。

```bash
pip3 install ultralytics
```

找到这个文件：~/.local/lib/python3.8/site-packages/ultralytics/nn/tasks.py。如果找不到，请通过pip3 show ultralytics查包的安装位置。
找到这个函数：
```python
def _predict_once(self, x, profile=False, visualize=False, embed=None):
    ...
    return x
```
修改返回值，加一个transpose操作，这样更有利于cpu后处理连续取数。将`return x`修改为：
```
    return x.permute(0, 2, 1)
```

## 2. 导出onnx模型
如果使用tpu-mlir编译模型，则必须先将Pytorch模型导出为onnx模型。YOL0v8官方仓库提供了模型导出接口，可以直接使用它导出onnx模型：

```python
from ultralytics import YOLO
  
model = YOLO("yolov8s-obb.pt")

model.export(format='onnx', opset=17, dynamic=True)
```

上述脚本会在原始pt模型所在目录下生成导出的onnx模型`yolov8s-obb.onnx`。
