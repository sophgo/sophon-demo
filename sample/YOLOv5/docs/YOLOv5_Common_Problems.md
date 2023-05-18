[简体中文](./YOLOv5_Common_Problems.md) | [English](./YOLOv5_Common_Problems_EN.md)

# YOLOv5移植常见问题
## 1. Anchor尺寸不符
如果出现检测框的中心位置与目标中心位置基本吻合，但是**检测框的尺寸明显不符**，如尺寸远大于目标，则预设的anchor尺寸与图片尺寸不一致。

这是因为YOLOv5在训练之前会检测预设anchor尺寸的有效性，如果不符合需求，则会根据训练集中标注的检测框的大小重新聚类生成合适anchor的尺寸。 因此，在移植算法进行推理时，**需要手动更改anchor尺寸，与训练前聚类生成的尺寸一致**。

通用操作方法：使用原始代码加载模型，设置断点，查看模型中的anchors参数。下面是提供用来查看开源仓库YOLOv5中权重的Anchor参数`python`代码

```python
import torch
from models.experimental import attempt_load
modelPath = 'Your_Model_Path'
model = attempt_load(modelPath, map_location=torch.device('cpu'))
m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]
print(f'小目标anchor : {m.anchors[0]*m.stride[0]}')
print(f'中目标anchor : {m.anchors[1]*m.stride[1]}')
print(f'大目标anchor : {m.anchors[2]*m.stride[2]}')
```

## 2. TPU-NNTC量化精度损失
使用Fp32 bmodel模型对模型进行推理，预测结果符合预期值；当使用自动量化工具对模型进行量化移植时，检测框的置信度、位置发生较大的变化，检测框的位置一般会**偏移至左上角**。

该问题由于量化设置的规则不合理导致，对于YOLOv5模型，一般**不对最后三个卷积层以及后续处理进行量化处理**，使用 `bmrt_test –bmodel YOUR_MODUL_PATH` 测试bmodel最后输出结果的类型，如果显示为INT8，则需要使用netron软件查看fp32的protext文件，确认最后三个卷积层的名字，冻结最后三层不量化。

> **查看权重名称的方法:**  
使用一键量化脚本后，会在原模型所在目录生成`yolov5s_bmnetp_test_fp32.prototxt`，需要下载该文件至本地，使用Netron打开该protext文件。其中，Netron为神经网络可视化软件，可以使用[网页端](https://netron.app/)，查看神经网络计算图。找到网络输出层前面的最后三个卷积层，点击对应参数的按钮，在netron右侧会显示对应层的名称。

当使用上述操作后，精度损失依然较大，可以尝试使用以下策略：

1. 对于一键量化脚本的用户，我们提供了一种自动搜索量化策略，`--postprocess_and_calc_score_class detect_accuracy`，其中可选参数命令有`detect_accuracy`、`topx_accuracy_for_classify`、`feature_similarity`、`None`，注意该选项耗时较长
2. 如果对模型架构设计较为熟悉，可以使用 `--try_cali_accuracy_opt` 手动设计量化策略，可选量化策略均来自`calibration_use_pb`模块，可以参考`calibration_use_pb –help`
3. 一键量化脚本的精度或者灵活性不满足要求时，可以使用分步量化的方式，对量化过程中每一步使用的策略进行定制，具体方法可参考nntc参考手册

更多精度对齐的方法可以参考[精度对齐指导](../../../docs/FP32BModel_Precise_Alignment.md)。

## 3. 其他问题汇总
1. 前处理未对齐，例如yolov5采用灰边填充（letter box）的方式对图片进行放缩，不能直接resize；
2. 如果使用的是单输出的YOLOv5模型，解码部分对应的计算层不能量化；
3. 编译bmodel时，应当指定`--target`参数与将要部署的设备一致(BM1684/BM1684X)；
4. 如果无法正常加载模型，尝试使用`bm-smi`查看tpu状态以及利用率，如果产生异常(tpu状态为Fault或利用率100%)，请联系技术支持或者在GitHub上创建issue。