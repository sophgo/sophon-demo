# YOL0v8/YOLOv11模型导出
## 1. 准备工作
可选择从[YOLOv8官方主页](https://github.com/ultralytics/ultralytics?tab=readme-ov-file)下载yolov8s.pt模型，或在[导出onnx模型](#2-导出onnx模型)中自动下载模型。
安装如下依赖。

```bash
pip3 install ultralytics
pip3 install ultralytics==8.3.12 #如果需要导出yolov11，需要新版本ultralytics，且需要python版本>=3.10。
```

找到这个文件：~/.local/lib/python3.8/site-packages/ultralytics/nn/tasks.py。如果找不到，请通过pip3 show ultralytics查包的安装位置。
找到这个函数：
```python
def _predict_once(self, x, profile=False, visualize=False, embed=None):
    ...
    return x
```
修改返回值，加一个transpose操作，这样更有利于cpu后处理连续取数。将`return x`修改为：
```python
    return x.permute(0, 2, 1)
```

## 2. 导出onnx模型
如果使用tpu-mlir编译模型，则必须先将Pytorch模型导出为onnx模型。YOL0v8官方仓库提供了模型导出接口，可以直接使用它导出onnx模型：

```python
from ultralytics import YOLO
model = YOLO("yolov8s.pt")
model.export(format='onnx', opset=17, dynamic=True)
model = YOLO("yolo11s.pt")
model.export(format='onnx', opset=17, dynamic=True)
```

上述脚本会在原始pt模型所在目录下生成导出的`yolov8s.onnx`等模型。


# YOLOv9模型导出
## 1. 准备工作
YOLOv9模型导出是在Pytorch模型的生产环境下进行的，需提前根据[​YOLOv9官方开源仓库](https://github.com/WongKinYiu/yolov9)的要求安装好Pytorch环境，准备好相应的代码和模型，并保证模型能够在Pytorch环境下正常推理运行。

找到这个文件：`yolov9/models/yolo.py`。
找到这个类：
```python
class BaseModel(nn.Module):
    # YOLO base model
    def forward(self, x, profile=False, visualize=False):
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x
```
修改返回值，加一个transpose操作，这样更有利于cpu后处理连续取数。将`return x`修改为：
```python
        return x.permute(0, 2, 1)
```

## 2. 导出onnx模型

如果使用tpu-mlir编译模型，则必须先将Pytorch模型导出为onnx模型。YOLOv9官方仓库提供了模型导出接口，可以直接使用它导出onnx模型：

```bash
python3 export.py --weights yolov9-s-converted.pt --dynamic --include onnx
```

上述脚本会在原始pt模型所在目录下生成导出的onnx模型`yolov9-s-converted.onnx`。

# YOL0v12模型导出
## 1. 准备工作
YOLOv12模型导出是在Pytorch模型的生产环境下进行的，需提前根据[​YOLOv12官方开源仓库](https://github.com/sunsmarterjie/yolov12)的要求安装好Pytorch环境，准备好相应的代码和模型，并保证模型能够在Pytorch环境下正常推理运行。请先从官方主页下载`yolov12s.pt`模型。

找到这个文件：`yolov12/ultralytics/nn/tasks.py`。
找到这个函数：
```python
def _predict_once(self, x, profile=False, visualize=False, embed=None):
    ...
    return x
```
修改返回值，加一个transpose操作，这样更有利于cpu后处理连续取数。将`return x`修改为：
```python
    return x.permute(0, 2, 1)
```

## 2. 导出onnx模型
如果使用tpu-mlir编译模型，则必须先将Pytorch模型导出为onnx模型。YOL0v12官方仓库提供了模型导出接口，可以直接使用它导出onnx模型：
请在yolov12仓库文件夹内执行以下python代码
```python
from ultralytics import YOLO
model = YOLO("yolov12s.pt")
model.export(format="onnx", batch=1, opset=12, dynamic=True)
```

上述脚本会在原始pt模型所在目录下生成导出的`yolov12s.onnx`等模型。
