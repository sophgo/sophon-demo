# Python例程

## 目录

- [Python例程](#python例程)
  - [目录](#目录)
  - [1. 环境准备](#1-环境准备)
    - [1.1 x86/arm PCIe平台](#11-x86arm-pcie平台)
    - [1.2 SoC平台](#12-soc平台)
  - [2. 推理测试](#2-推理测试)
  - [2.1 参数说明](#21-参数说明)
    - [2.2 测试图片](#22-测试图片)

python目录下提供了一系列Python例程，具体情况如下：

| 序号   | Python例程      | 说明                                |
| ---- | ---------------- | -----------------------------------  |
| 1    | lightstereo_opencv.py | 使用OpenCV解码、OpenCV前处理、SAIL推理 |
| 2    | lightstereo_bmcv.py   | 使用SAIL解码、BMCV前处理、SAIL推理     |

## 1. 环境准备
### 1.1 x86/arm PCIe平台

如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon、sophon-opencv、sophon-ffmpeg和sophon-sail，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。

此外您可能还需要安装其他第三方库：
```bash
pip3 install opencv-python-headless
```

### 1.2 SoC平台

如果您使用SoC平台（如SE、SM系列边缘设备），并使用它测试本例程，刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包。您还需要交叉编译安装sophon-sail，具体可参考[交叉编译安装sophon-sail](../../../docs/Environment_Install_Guide.md#42-交叉编译安装sophon-sail)。

此外您可能还需要安装其他第三方库：
```bash
pip3 install opencv-python-headless
```

> **注:**
>
> 上述命令安装的opencv是公版opencv，如果您希望使用sophon-opencv，可以设置如下环境变量：
> ```bash
> export PYTHONPATH=$PYTHONPATH:/opt/sophon/sophon-opencv-latest/opencv-python/
> ```
> **若使用sophon-opencv需要保证python版本小于等于3.8。**

## 2. 推理测试
python例程不需要编译，可以直接运行，PCIe平台和SoC平台的测试参数和运行方式是相同的。

## 2.1 参数说明
lightstereo_opencv.py和lightstereo_bmcv.py的命令参数相同，以lightstereo_opencv.py的推理为例，参数说明如下：

```bash
usage:lightstereo_opencv.py [--input INPUT] [--bmodel BMODEL] [--dev_id DEV_ID]
--input:  要推理的数据集对应的文本文件路径
--bmodel: 用于推理的bmodel路径，默认使用stage 0的网络进行推理；
--dev_id: 用于推理的tpu设备id。
```

### 2.2 测试图片

图片测试实例如下：
```bash
# 测试整个文件夹
cd python
python3 lightstereo_opencv.py --input ../datasets/KITTI12/kitti12_train194.txt --bmodel ../models/BM1684X/LightStereo-S-SceneFlow_fp16_1b.bmodel --dev_id 0
```
测试结束后，会将预测结果保存在`results/images`下，同时会打印推理时间等信息。这里输出的是灰度图，如果需要彩色图，用户可以自行使用opencv的applyColorMap功能进行染色。
结果如图所示：
![000006_10](../pics/000006_10.png)