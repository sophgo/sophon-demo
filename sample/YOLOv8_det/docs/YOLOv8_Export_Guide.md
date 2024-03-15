# YOL0v8模型导出
## 1. 准备工作
YOL0v8模型导出是在Pytorch模型的生产环境下进行的，需提前根据[​YOLOv8官方开源仓库](https://github.com/ultralytics/ultralytics)的要求安装好Pytorch环境，准备好相应的代码和模型，并保证模型能够在Pytorch环境下正常推理运行。本例程导出环境版本为：`torch==2.0.1+cpu, onnx==1.15.0`。

同时需要安装官方提供的第三方库：
```bash
pip3 install ultralytics==8.1.27
```

## 2. 导出onnx模型
如果使用tpu-mlir编译模型，则必须先将Pytorch模型导出为onnx模型。YOL0v8官方仓库提供了模型导出接口，可以直接使用它导出onnx模型：

```python
from ultralytics import YOLO
# Load a model
model = YOLO("yolov8s.pt")  # 载入预训练模型
# Export the model
success = model.export(format="onnx", dynamic=True)  # 导出动态ONNX模型
```

上述脚本会在原始pt模型所在目录下生成导出的onnx模型`yolov8s.onnx`。

为了加快cpp例程后处理速度，这里在源yolov8的输出层后再接一个transpose层，这样更加适合cpp例程后处理时连续取数，这种方式称之为yolov8s_opt。


需要安装官方提供的第三方库：
```bash
git clone https://github.com/ultralytics/ultralytics
cd ultralytics
```

修改源码 ultralytics/ultralytics/nn/tasks.py
```
def _predict_once(self, x, profile=False, visualize=False, embed=None)函数得返回值
# return x
return x.permute(0, 2, 1)
```
修改以上源码之后执行以上脚本会在原始pt模型所在目录下生成导出的onnx模型即为opt版本得`yolov8s.onnx`


## 4. 常见问题
TODO
