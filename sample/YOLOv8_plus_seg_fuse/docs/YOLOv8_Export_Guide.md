# YOLOv8模型导出
## 1. 准备工作
可选择从[YOLOv8官方主页](https://github.com/ultralytics/ultralytics?tab=readme-ov-file)下载yolov8s-seg.pt模型，或在[导出onnx模型](#2-导出onnx模型)中自动下载模型。
安装如下依赖。

```bash
pip3 install ultralytics --force-reinstall 
pip3 install onnx
```

## 2. 导出onnx模型
如果使用tpu-mlir编译模型，则必须先将Pytorch模型导出为onnx模型。YOLOv8官方仓库提供了模型导出接口，可以直接使用它导出onnx模型：

```python
from ultralytics import YOLO
model = YOLO("yolov8s-seg.pt")
model.export(format='onnx', opset=17, dynamic=True)
```

上述脚本会在原始pt模型所在目录下生成导出的`yolov8s-seg.onnx`等模型。

## 3. 融合后处理（YOLOv8_plus_seg_fuse 专用）

`YOLOv8_plus_seg_fuse`例程在编译脚本（`scripts/gen_*bmodel_mlir.sh`）的`model_transform.py`中加入了`--add_postprocess yolov8_seg`，将box解码、置信度过滤、NMS、mask裁剪等后处理融合到TPU中计算，从而大幅减少主机端后处理耗时。因此：

- 按第2节导出标准的`yolov8s-seg.onnx`即可，融合后处理是在TPU-MLIR转换阶段自动加上的，无需对onnx做额外修改；
- 融合后处理对数值精度敏感，编译FP16/INT8模型时需配合`scripts`目录下的qtable（`yolov8s_seg_fuse_qtable`用于INT8、`yolov8s_seg_fuse_qtable_f16`用于FP16）将相关层设为F32，详见根目录README第3.2节。