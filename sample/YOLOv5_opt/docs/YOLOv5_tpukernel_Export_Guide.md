# YOLOv5模型导出
## 1. 准备工作
YOLOv5模型导出是在Pytorch模型的生产环境下进行的，需提前根据[​YOLOv5官方开源仓库](https://github.com/ultralytics/yolov5)的要求安装好Pytorch环境，准备好相应的代码和模型，并保证模型能够在Pytorch环境下正常推理运行。

## 2. 主要步骤
### 2.1 修改models/yolo.py

YOLOv5不同版本的代码导出的YOLOv5模型的输出会有所不同，根据不同的组合可能会有1、2、3、4个输出的情况，主要取决于model/yolo.py文件中的class Detect的forward函数。本例程使用的tpu_kernel中的yolo后处理接口，接收的输入是yolo最后三个卷积层的输出，所以在导出onnx模型的时候，需要把源代码相应的部分注释掉，如下所示：

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

### 2.2 导出torchscript模型
​Pytorch模型在编译前要经过`torch.jit.trace`，trace后的模型才能编译BModel。YOLOv5官方仓库提供了模型导出脚本`export.py`，可以直接使用它导出torchscript模型：

```bash
# 下述脚本可能会根据不用版本的YOLOv5有所调整，请以官方仓库说明为准
python3 export.py --weights ${PATH_TO_YOLOV5S_MODEL}/yolov5s.pt --include torchscript
```

上述脚本会在原始pt模型所在目录下生成导出的torchscript模型，导出后可以修改模型名称以区分不同版本和输出类型，如`yolov5s_tpukernel.pt`表示带有3个卷积输出的JIT模型。

**注意：** 导出的模型建议以`.pt`为后缀，以免在后续模型编译量化中发生错误。

### 2.3 导出onnx模型
如果使用tpu-mlir编译模型，则必须先将Pytorch模型导出为onnx模型。YOLOv5官方仓库提供了模型导出脚本`export.py`，可以直接使用它导出onnx模型：

```bash
# 下述脚本可能会根据不用版本的YOLOv5有所调整，请以官方仓库说明为准
python3 export.py --weights ${PATH_TO_YOLOV5S_MODEL}/yolov5s.pt --include onnx --dynamic
```

上述脚本会在原始pt模型所在目录下生成导出的onnx模型，导出后可以修改模型名称以区分不同版本和输出类型，如`yolov5s_tpukernel.onnx`表示带有3个卷积输出的onnx模型。

## 3. 常见问题

版本较新的yolov5源码改了model.py之后运行export.py可能会出现这种报错：
```bash
Traceback (most recent call last):
  File "/workspace/open-source/yolov5/export.py", line 940, in <module>
    main(opt)
  File "/workspace/open-source/yolov5/export.py", line 935, in main
    run(**vars(opt))
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
  File "/workspace/open-source/yolov5/export.py", line 822, in run
    shape = tuple((y[0] if isinstance(y, tuple) else y).shape)  # model output shape
AttributeError: 'list' object has no attribute 'shape'
```
只需要把export.py大约822行左右的这两行代码注释掉即可，然后重新export：
```bash
# shape = tuple((y[0] if isinstance(y, tuple) else y).shape)  # model output shape
metadata = {"stride": int(max(model.stride)), "names": model.names}  # model metadata
# LOGGER.info(f"\n{colorstr('PyTorch:')} starting from {file} with output shape {shape} ({file_size(file):.1f} MB)")
```