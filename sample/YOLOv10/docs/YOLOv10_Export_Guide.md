# YOL0v8模型导出
## 1. 准备工作
YOL0v8模型导出是在Pytorch模型的生产环境下进行的，需提前根据[​YOLOv8官方开源仓库](https://github.com/THU-MIG/yolov10)的要求安装好Pytorch环境，准备好相应的代码和模型，并保证模型能够在Pytorch环境下正常推理运行。本例程导出环境版本为：`torch==2.0.1+cpu, onnx==1.15.0`。


## 2. 导出onnx模型
如果使用tpu-mlir编译模型，则必须先将Pytorch模型导出为onnx模型。YOL0v10官方仓库提供了模型导出接口，可以直接使用它导出onnx模型：

```python
from ultralytics import YOLO
# Load a model
model = YOLO("yolov10s.pt")  # 载入预训练模型
# Export the model
success = model.export(format="onnx",opset=13,dynamic=True)  # 导出静态ONNX模型，需设置batch参数注意opset为12/13
```


由于目前mlir不支持mod操作，所以需要下载源码，修改其后处理yolov10/ultralytics/utils/ops.py/v10postprocess：
```bash
git clone https://github.com/THU-MIG/yolov10.git
cd ultralytics
```

```
floored_division = torch.floor_divide(index, nc)
mod_result = index - nc * floored_division

labels = mod_result
index = floored_division

``

## 4. 常见问题
TODO
