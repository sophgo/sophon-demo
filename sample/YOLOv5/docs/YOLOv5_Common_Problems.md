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

## 2. 其他问题汇总
1. 前处理未对齐，例如yolov5采用灰边填充（letter box）的方式对图片进行放缩，不能直接resize；
2. 如果使用的是单输出的YOLOv5模型，解码部分对应的计算层不能量化；
3. 编译bmodel时，应当指定`--target`参数与将要部署的设备一致(BM1684/BM1684X)；
4. 如果无法正常加载模型，尝试使用`bm-smi`查看tpu状态以及利用率，如果产生异常(tpu状态为Fault或利用率100%)，请联系技术支持或者在GitHub上创建issue。
5. 精度对齐的方法可以参考[精度对齐指导](../../../docs/FP32BModel_Precise_Alignment.md)。