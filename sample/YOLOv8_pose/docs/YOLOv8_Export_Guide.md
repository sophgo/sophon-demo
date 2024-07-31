# YOL0v8模型导出
## 1. 准备工作
可选择从[YOLOv8-pose官方主页](https://github.com/ultralytics/ultralytics/issues/1915)下载YOL0v8-pose.pt模型，或在[导出onnx模型](#2-导出onnx模型)中自动下载模型。
安装如下依赖。

```bash
pip3 install ultralytics==8.1.27
```

## 2. 导出onnx模型
如果使用tpu-mlir编译模型，则必须先将Pytorch模型导出为onnx模型。YOL0v8官方仓库提供了模型导出接口，可以直接使用它导出onnx模型：

```python
from ultralytics import YOLO
  
# 加载pt模型
yolov8s_pose = YOLO('yolov8s-pose.pt') #方式一，会自动从官网下载yolov8s-pose.pt模型
yolov8s_pose = YOLO('/path/to/yolov8s-pose.pt') #方式二，指定本地模型路径

# 导出opset17标准的动态ONNX模型
yolov8s_pose.export(format='onnx', opset=17, dynamic=True) 
```

上述脚本会在原始pt模型所在目录下生成导出的onnx模型`yolov8s-pose.onnx`。
