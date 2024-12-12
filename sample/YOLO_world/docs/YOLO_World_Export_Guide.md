# YOLO_world模型导出
## 1. 准备工作
可选择从[YOLO_world官方](https://github.com/ultralytics/assets/releases/tag/v8.3.0)下载YOLOv8s-worldv2.pt模型，或在[导出onnx模型](#2-导出onnx模型)中自动下载模型。
安装如下依赖。

```bash
pip3 install ultralytics onnx onnxsim
```

## 2. 导出onnx模型
如果使用tpu-mlir编译模型，则必须先将Pytorch模型导出为onnx模型。使用如下脚本，可以直接使用它导出onnx模型：

```python
from ultralytics import YOLOWorld
from copy import deepcopy
import torch

class ModelExporter(torch.nn.Module):
    def __init__(self, yoloModel, device='cpu'):
        super(ModelExporter, self).__init__()
        model = yoloModel.model
        model = deepcopy(model).to(device)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.float()
        model = model.fuse()

        self.model = model
        self.device = device

    def forward(self, x, txt_feats):
        return self.model.predict(x, txt_feats=txt_feats)

    def export(self, output_dir, model_name, img_width, img_height, num_classes):
        x = torch.randn(1, 3, img_height, img_width, requires_grad=False).to(self.device)
        txt_feats = torch.randn(1, num_classes, 512, requires_grad=False).to(self.device)

        print(x.shape, txt_feats.shape)

        # Export model
        output_path = f"{output_dir}/{model_name}.onnx"
        with torch.no_grad():
            torch.onnx.export(self,
                              (x, txt_feats),
                              output_path,
                              do_constant_folding=True,
                              opset_version=12,
                              input_names=["images", "txt_feats"],
                              output_names=["output"],
                              dynamic_axes={
                              "images": {0: "batch_size"} 
                          })


model_name = 'yolov8s-worldv2' #@param ["yolov8s-worldv2", "yolov8m-worldv2", "yolov8l-worldv2", "yolov8x-worldv2"]
input_width = 640 #@param {type:"slider", min:32, max:4096, step:32}
input_height = 640 #@param {type:"slider", min:32, max:4096, step:32}
num_classes = 80 # @param {type:"integer"}

yoloModel = YOLOWorld(model_name)
yoloModel.set_classes([""] * num_classes)

# Initialize model exporter
export_model = ModelExporter(yoloModel)

# Export model
export_model.export(".", model_name, input_width, input_height, num_classes)

# Simplify
!onnxsim {model_name}.onnx yoloworld.onnx

```

上述脚本会在原始pt模型所在目录下生成导出的onnx模型`yoloworld.onnx`。