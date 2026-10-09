# YOLO26-pose模型导出
## 1. 准备工作
YOLO26-pose模型导出是在Pytorch模型的生产环境下进行的，需提前根据[​YOLO26官方开源仓库](https://github.com/ultralytics/ultralytics)的要求安装好环境，准备好相应的代码和模型，并保证模型能够在Pytorch环境下正常推理运行。本例程导出环境版本为：`torch-2.4.1, onnx 1.17.0, ultralytics-8.4.105`。


## 2. 导出onnx模型
如果使用tpu-mlir编译模型，则必须先将Pytorch模型导出为onnx模型。YOLO26官方仓库提供了模型导出接口，可以直接使用它导出onnx模型，应设置opset为13。

```python
from ultralytics import YOLO
# Load a model
model = YOLO("yolo26s-pose.pt")
# Export the model，导出opset13标准的动态ONNX模型
model.export(format="onnx", opset=13, dynamic=True)  # 注意：必须使用dynamic=True
```

上述脚本会在原始pt模型所在目录下生成导出的onnx模型`yolo26s-pose.onnx`。

**导出时需要使用`dynamic=True`**：YOLO26-pose的注意力分支和多尺度检测头中存在依赖batch维度的`Reshape`算子，若按静态shape导出（默认`dynamic=False`），这些`Reshape`的目标形状会被固化并写入常量`1`，导致后续只能编译batch_size=1的BModel。使用`dynamic=True`时batch维度会被导出为动态维度，才可以编译出batch_size=4的BModel。

## 3. 模型输出说明
YOLO26-pose默认使用端到端（end2end）输出，不需要NMS后处理，导出的onnx模型输入输出如下：

- 输入：`images`，shape为`[batch, 3, 640, 640]`（batch为动态维度），float32，RGB、归一化到`[0, 1]`。
- 输出：`output0`，shape为`[batch, 300, 57]`，float32，共300个人体候选框，已按置信度从高到低排序。每一行的布局为：

```
[x1, y1, x2, y2, score, class_id, kpt0_x, kpt0_y, kpt0_score, ..., kpt16_x, kpt16_y, kpt16_score]
```

其中`[x1, y1, x2, y2]`和17个关键点坐标都是相对于letterbox后640x640图像的坐标，关键点数量为17、分值范围`[0, 1]`；人体检测只有一个类别`person`，所以`class_id`恒为0。

例程中的后处理（`python/postprocess_numpy.py`、`cpp/yolo26_pose_bmcv/yolo26_pose.cpp`）按照上述布局解析输出，并将坐标反映射回原图。
