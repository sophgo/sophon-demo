# SAM2模型编译

## 1.准备工作
SAM模型导出是在Pytorch模型的生产环境下进行的，需提前根据[​SAM官方开源仓库](https://github.com/facebookresearch/segment-anything-2)的要求安装好Pytorch环境，准备好相应的代码和模型，并保证模型能够在Pytorch环境下正常推理运行。

## 导出onnx
onnx文件的导出可参考：[samexporter](https://github.com/vietanhdev/samexporter.git)
## 编译bmodel
编译sam2_encoder bmodel时会出现两处算子不支持：`Mod`和`Size`

`model_transform.py --model_name sam2_encoder --model_def ../../sam2_hiera_tiny_encoder.onnx --input_shapes [[1,3,1024,1024]] --output_name high_res_feats_0,high_res_feats_1,image_embed --mlir sam2_encoder.mlir`

```
2024/08/06 14:43:59 - INFO : TPU-MLIR v1.8.beta.0-248-gd0cbae79f-20240802
2024/08/06 14:43:59 - INFO :
         _____________________________________________________
        | preprocess:                                           |
        |   (x - mean) * scale                                  |
        '-------------------------------------------------------'
  config Preprocess args :
        resize_dims           : same to net input dims
        keep_aspect_ratio     : False
        keep_ratio_mode       : letterbox
        pad_value             : 0
        pad_type              : center
        --------------------------
        mean                  : [0.0, 0.0, 0.0]
        scale                 : [1.0, 1.0, 1.0]
        --------------------------
        pixel_format          : bgr
        channel_format        : nchw

2024/08/06 14:43:59 - INFO : Input_shape assigned
WARNING: onnx model check failed
2024/08/06 14:44:00 - INFO : WARNING: ConstantFolding failed.
2024/08/06 14:44:00 - INFO : ConstantFolding finished
2024/08/06 14:44:00 - INFO : skip_fuse_bn:False
2024/08/06 14:44:00 - INFO : WARNING: onnxsim opt failed.
2024/08/06 14:44:00 - INFO : Onnxsim opt finished
WARNING: onnx model check failed
2024/08/06 14:44:00 - INFO : WARNING: ConstantFolding failed.
2024/08/06 14:44:00 - INFO : ConstantFolding finished
Traceback (most recent call last):
  File "/workspace/tpu-mlir/python/tools/model_transform.py", line 350, in <module>
    tool.model_transform(args.mlir, args.add_postprocess, args.patterns_count)
  File "/workspace/tpu-mlir/python/tools/model_transform.py", line 57, in model_transform
    self.converter.generate_mlir(mlir_origin)
  File "/workspace/tpu-mlir/python/transform/OnnxConverter.py", line 721, in generate_mlir
    raise RuntimeError("Op not support:{}".format(unsupported))
RuntimeError: Op not support:{'Mod', 'Size'}
```
 Mod算子
**出现位置：** sam2/modeling/backbones
/utils.py
```
def window_partition(x, window_size):
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.
    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows, (Hp, Wp)
```

代码中使用了取模运算 % 来计算 pad_h 和 pad_w，这是为了确定需要多少填充来使得高度 H 和宽度 W 能够被 window_size 整除。
**替代方法：**
```python     
pad_h = ((H + window_size - 1) // window_size) * window_size - H
pad_w = ((W + window_size - 1) // window_size) * window_size - W
```
通过计算 (H + window_size - 1) // window_size 来获取在不使用 % 运算的情况下，高度和宽度分别需要多少个 window_size 单位。然后，将这个数值乘以 window_size 并从原始尺寸中减去，得到需要的填充量 pad_h 和 pad_w。

Size算子**出现位置：** sam2/modeling/backbones/hieradet.py：269
```python
pos_embed = pos_embed + window_embed.tile(
    [x // y for x, y in zip(pos_embed.shape, window_embed.shape)]
)
```
这个操作出了出现Size算子，还出现了Sub、Greater、Less 和 If 算子

Size：获取张量的某个维度的大小。在PyTorch中，这可以通过 .size() 或 .shape 属性获得。

Sub：执行张量之间的减法操作。这可能是用来计算 tile 函数中的重复次数，即 pos_embed.shape 和 window_embed.shape 之间的差异。

Greater 和 Less：这些算子用于比较操作，可能是用来检查 pos_embed 和 window_embed 的形状，以确保平铺操作可以正确执行。比如，确保 pos_embed 的每个维度都大于或等于 window_embed 的对应维度。

If：这是一个条件执行算子，在ONNX中用于执行基于条件的分支。在这个上下文中，If 简单地可以用来决定是否执行某个操作，比如当 pos_embed 和 window_embed 的形状不匹配时，可能需要先调整形状再进行平铺操作。

这些算子的出现表明在转换过程中，ONNX转换器在处理 tile 操作和形状计算时，需要进行更复杂的形状和大小检查。这可能是因为ONNX的平铺操作（Tile）对输入的形状有特定的要求，或者是为了确保转换后的模型能够处理不同的输入形状，而PyTorch中的代码可能只是假设了特定的输入形状。

**替代方法：**
```python
window_embed = window_embed.repeat(1, 1, 32, 32)
pos_embed = pos_embed + window_embed
```
预先计算平铺次数： pos_embed 和 window_embed 的形状在模型运行前是已知的，此处直接计算出平铺次数，关闭在模型运行时动态计算，这样在转换到ONNX时就不会出现 Size 和相关算子。

修改之后再次运行就可以正常编译

