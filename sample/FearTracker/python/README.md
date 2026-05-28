# Python例程

## 目录

- [Python例程](#python例程)
  - [目录](#目录)
  - [1. 环境准备](#1-环境准备)
    - [1.1 x86/arm/riscv PCIe平台](#11-x86armriscv-pcie平台)
    - [1.2 SoC平台](#12-soc平台)
  - [2. 推理测试](#2-推理测试)
    - [2.1 参数说明](#21-参数说明)
    - [2.2 使用方式](#22-使用方式)
    - [2.3 测试视频](#23-测试视频)
  - [3. 性能测试](#3-性能测试)

python目录下提供了一系列Python例程，具体情况如下：

| 序号 |  Python例程       | 说明          |
| ---- | ---------------- | ------------- |
| 1    | fear_tracker.py  | 使用SAIL推理  |

## 1. 环境准备
### 1.1 x86/arm/riscv PCIe平台

如果您在x86/arm/riscv平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon、sophon-sail，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)或[riscv-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#6-riscv-pcie平台的开发和运行环境搭建)。

此外您可能还需要安装其他第三方库：
```bash
pip3 install numpy opencv-python-headless
```

### 1.2 SoC平台

如果您使用SoC平台（如SE、SM系列边缘设备），并使用它测试本例程，刷机后在`/opt/sophon/`下已经预装了相应的libsophon运行库包。您还需要交叉编译安装sophon-sail，具体可参考[交叉编译安装sophon-sail](../../../docs/Environment_Install_Guide.md#42-交叉编译安装sophon-sail)。

此外您可能还需要安装其他第三方库：
```bash
pip3 install numpy opencv-python-headless
```

## 2. 推理测试
python例程不需要编译，可以直接运行，PCIe平台和SoC平台的测试参数和运行方式是相同的。

### 2.1 参数说明
```bash
usage: fear_tracker.py [--bmodel BMODEL] [--input INPUT]
                        [--initial_bbox BBOX] [--output OUTPUT]
                        [--dev_id DEV_ID]

--bmodel: 用于推理的bmodel路径，默认使用 ../models/BM1684X/feartracker_fp32_1b.bmodel；
--input: 输入视频文件路径，必填；
--initial_bbox: 初始跟踪目标边界框，格式为 x,y,w,h（如 163,53,45,174），必填；
--output: 输出视频文件路径，可选，指定后会将跟踪结果可视化保存；
--dev_id: 用于推理的tpu设备id，默认为0；
```

### 2.2 使用方式
模型加载使用SAIL的`sail.Engine`类，采用`SYSIO`模式（numpy数组输入输出）。每帧处理流程为：
1. 读取视频帧，根据上一帧的跟踪框裁剪搜索区域（256x256）并做归一化预处理
2. 初始帧裁剪模板区域（128x128），保存模板图像供后续帧复用
3. 将模板图像和搜索图像送入TPU推理，得到bbox_pred和cls_pred
4. 后处理：sigmoid→逐元素分类得分→argmax→解码回归坐标→缩放回原始图像坐标系

```python
import sophon.sail as sail

# 加载模型
engine = sail.Engine(bmodel_path, dev_id, sail.IOMode.SYSIO)
graph_name = engine.get_graph_names()[0]

# 逐帧推理
input_data = {"template": template_img, "search": search_img}
output = engine.process(graph_name, input_data)
bbox_pred = output["bbox_pred_Exp"]   # [1, 4, 16, 16]
cls_pred = output["cls_pred_Mul"]     # [1, 1, 16, 16]
```

### 2.3 测试视频
视频测试实例如下：
```bash
cd python
python3 fear_tracker.py \
    --bmodel ../models/BM1684X/feartracker_fp32_1b.bmodel \
    --input ../datasets/test.mp4 \
    --initial_bbox 163,53,45,174 \
    --output ../results/output.mp4
```

BM1688平台测试实例如下：
```bash
cd python
python3 fear_tracker.py \
    --bmodel ../models/BM1688/feartracker_bm1688_fp16_1b.bmodel \
    --input ../datasets/test.mp4 \
    --initial_bbox 163,53,45,174 \
    --output ../results/output.mp4
```

测试结束后，输出每帧的跟踪边界框位置，如指定了`--output`参数，会保存带跟踪框的可视化视频。

## 3. 性能测试

在不同的测试平台上，使用不同的模型进行测试，性能测试结果如下：

|    测试平台  |                  测试模型                       | 帧率  | 平均推理时间(ms) |
| ----------- | ---------------------------------------------- | ----- | ---------------- |
|   SE9-16    | BM1688/feartracker_bm1688_fp16_1b.bmodel       |  64   |      15.5        |

> **测试说明**：
> 1. 测试视频：661帧，分辨率 640x360；
> 2. 平均推理时间包括模板/搜索图像裁剪、numpy预处理、TPU推理、后处理（sigmoid→argmax→bbox解码→坐标缩放）的完整流程；
> 3. 模板图像仅在首帧预处理一次，后续帧复用，不计入后续帧的推理时间；
> 4. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 5. SE9-16的主控处理器为8核CA55@1.6GHz，BM1688 TPU；