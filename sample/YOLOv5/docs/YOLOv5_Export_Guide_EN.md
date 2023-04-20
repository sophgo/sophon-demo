[简体中文](./YOLOv5_Export_Guide.md) | [English](./YOLOv5_Export_Guide_EN.md)

# YOLOv5 Model Derivation
## 1. Preparatory Work
YOLOv5 model is derived under the environment of Pytorch model of production, advanced according to the required [YOLOv5 official open source warehouse](https://github.com/ultralytics/yolov5), the requirements of the installed Pytorch environment, ready for the corresponding code and model, It also ensures that the model can run properly under Pytorch environment.
> **Attention**:recommend to use the torch vision of`1.8.0+cpu`to avoid model compilation failures due to the pytorch version.

## 2. Main Steps
### 2.1 Modify models/yolo.py

The output of the YOLOv5 model will be different for different versions of the YOLOv5 code. Depending on the combination, there may be 1, 2, 3, or 4 outputs, depending on the class Detect forward function in the Model/olo. It is recommended to modify the last return statement of the forward function of the Detect class to achieve one or three outputs. If the model is output 3, sigmoid and predicted coordinates need to be converted in post-processing.

```python
    ....
    
    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))
                
        # return x if self.training else (torch.cat(z, 1), x)  # 4 outputs
        return x if self.training else x                       # 3 outputs
        # return x if self.training else (torch.cat(z, 1))     # 1 output
        ....
```

### 2.2 Export the Torchscript Model
The Pytorch model must go through `torch.jit.trace` before compilation, and the model after trace can use tpu-nntc to compile BModel.YOLOv5 official repository provides model export scripts`export.py`,you can use it directly to export the torchscript model:

```bash
# The following script may be adjusted based on different versions of YOLOv5. Please refer to the official repository instructions
python3 export.py --weights ${PATH_TO_YOLOV5S_MODEL}/yolov5s.pt --include torchscript
```

The above script generates the exported torchscript model in the directory where the original pt model resides. After exporting, the model name can be changed to distinguish between different versions and output types,such as`yolov5s_v6.1_3output.torchscript.pt`represents a JIT model with three outputs.

**Attention:** The derived model is suggested to be suffixed with `.pt`,to avoid errors in subsequent model compilation quantization.

### 2.3 Derive the onnx Model
If you compile the model using tpu-mlir, you must first export the Pytorch model as an onnx model.YOLOv5 official repository provides the model export script ,you can use it directly to derive the onnx model:

```bash
# The following script may be adjusted based on different versions of YOLOv5. Please refer to the official repository instructions
python3 export.py --weights ${PATH_TO_YOLOV5S_MODEL}/yolov5s.pt --include onnx --dynamic
```

The script generates the exported onnx model in the directory where the original pt model resides. After the export, you can modify the model name to distinguish between different versions and output types.For example,`yolov5s_v6.1_3output.onnx`represents an onnx model with 3 outputs.

## 3. Common Problems
TODO
