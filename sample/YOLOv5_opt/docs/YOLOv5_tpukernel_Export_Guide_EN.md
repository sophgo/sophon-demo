# YOLOv5 export
<div align="center">

## [简体中文](./YOLOv5_tpukernel_Export_Guide.md) | [English](./YOLOv5_tpukernel_Export_Guide_EN.md)
</div>

## 1. preparation
YOLOv5 model export is carried out in the production environment of Pytorch models, you need to install the Pytorch environment in advance according to the requirements of [YOLOv5 official repo](https://github.com/ultralytics/yolov5), prepare the corresponding code and model, and ensure that the model can run normally in the Pytorch environment.
> **Note**: 
It is recommended to use the Torch version of '1.8.0+CPU' to avoid failure of the TPU-NNTTC model compilation due to the PyTorch version.

## 2. main step
### 2.1 modify models/yolo.py

The yolo post-processing interface in the tpu_kernel used in this example receives the output of the last three convolutional layers of yolo, so when exporting the onnx model, you need to comment out the corresponding part of the source code, as follows:

```python
    ....
    
    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            # bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            # x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            # if not self.training:  # inference
            #     if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
            #         self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

            #     y = x[i].sigmoid()
            #     if self.inplace:
            #         y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
            #         y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            #     else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
            #         xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
            #         wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            #         y = torch.cat((xy, wh, y[..., 4:]), -1)
            #     z.append(y.view(bs, -1, self.no))
                
        return x
        ....
```

### 2.2 Export Torchscript models
The Pytorch model must go through `torch.jit.trace` before compiling, and the traced model can be compiled BModel using tpu-nntc. The official YOLOv5 repository provides the model export script `export.py`, which can be used directly to export torchscript models:

```bash
# The following script may be adjusted according to the different version of YOLOv5, please refer to the official repository description
python3 export.py --weights ${PATH_TO_YOLOV5S_MODEL}/yolov5s.pt --include torchscript
```

The above script will generate the exported torchscript model in the same directory as the original pt model, and after exporting, you can modify the model name to distinguish different versions and output types, such as `yolov5s_tpukernel.pt` to represent a JIT model with 3 convolution outputs.

Note: It is recommended that exported models be suffixed with '.pt' to avoid errors in subsequent model compilation quantization.

### 2.3 Export ONNX models
If you compile the model with tpu-mlir, you must first export the Pytorch model as an onnx model. The official YOLOv5 repository provides a model export script 'export.py', which can be used directly to export onnx models:

```bash
# The following script may be adjusted according to the different version of YOLOv5, please refer to the official repository description
python3 export.py --weights ${PATH_TO_YOLOV5S_MODEL}/yolov5s.pt --include onnx --dynamic
```

The above script will generate the exported ONNX model in the same directory as the original PT model, and after exporting, you can modify the model name to distinguish different versions and output types, such as `yolov5s_tpukernel.onnx` to represent the ONNX model with 3 convolution outputs.

## 3. frequently asked questions
TODO
