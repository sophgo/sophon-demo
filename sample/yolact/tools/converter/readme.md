# 原始模型转换

将原始模型转换为torchscript.trace后的JIT模型或ONNX模型。

## 模型参数

| backbone      | input shape               | input                              | mode                                             | cfg                |
| ------------- | ------------------------- | ---------------------------------- | ------------------------------------------------ | ------------------ |
| Resnet50-FPN  | (batch_size, 3, 550, 550) | yolact_base_54_800000模型路径      | `tstrace`表示导出JIT模型，`onnx`表示导出ONNX模型 | `yolact_resnet50`  |
| Darknet53-FPN | (batch_size, 3, 550, 550) | yolact_darknet53_54_800000模型路径 | `tstrace`表示导出JIT模型，`onnx`表示导出ONNX模型 | `yolact_darknet53` |
| Resnet101-FPN | (batch_size, 3, 550, 550) | yolact_resnet50_54_800000模型路径  | `tstrace`表示导出JIT模型，`onnx`表示导出ONNX模型 | `yolact_base`      |
| Resnet101-FPN | (batch_size, 3, 700, 700) | yolact_im700_54_800000模型路径     | `tstrace`表示导出JIT模型，`onnx`表示导出ONNX模型 | `yolact_im700`     |

### 转换JIT模型或ONNX模型

以下命令以转换JIT模型为例，若需要转换ONNX模型，只需要将`mode`参数改成`onnx`即可。

```bash
# yolact_base_54_800000模型
python3 ./convert.py --input yolact_base_54_800000.pth --mode tstrace --cfg yolact_base
# yolact_darknet53_54_800000模型
python3 ./convert.py --input yolact_darknet53_54_800000.pth --mode tstrace --cfg yolact_darknet53
# yolact_resnet50_54_800000模型
python3 ./convert.py --input yolact_resnet50_54_800000.pth --mode tstrace --cfg yolact_resnet50
# yolact_im700_54_800000模型
python3 ./convert.py --input yolact_im700_54_800000.pth --mode tstrace --cfg yolact_im700
```
