# Python例程

## 目录

* [1. 环境准备](#1-环境准备)
    * [1.1 x86/arm PCIe平台](#11-x86arm-pcie平台)
    * [1.2 SoC平台](#12-soc平台)
* [2. 推理测试](#2-推理测试)
    * [2.1 参数说明](#21-参数说明)
    * [2.2 测试图片](#21-测试图片)

python目录下提供了一系列Python例程，具体情况如下：

| 序号   | Python例程      | 说明                                |
| ---- | ---------------- | -----------------------------------  |
| 1    | resnet_opencv.py | 使用OpenCV解码、OpenCV前处理、SAIL推理 |
| 2    | resnet_bmcv.py   | 使用SAIL解码、BMCV前处理、SAIL推理     |

## 1. 环境准备
### 1.1 x86/arm PCIe平台

如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon、sophon-opencv、sophon-ffmpeg和sophon-sail，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。

此外您可能还需要安装其他第三方库：
```bash
pip3 install 'opencv-python-headless<4.3'
```

### 1.2 SoC平台

如果您使用SoC平台（如SE、SM系列边缘设备），并使用它测试本例程，刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包。您还需要交叉编译安装sophon-sail，具体可参考[交叉编译安装sophon-sail](../../../docs/Environment_Install_Guide.md#42-交叉编译安装sophon-sail)。

此外您可能还需要安装其他第三方库：
```bash
pip3 install 'opencv-python-headless<4.3'
```

## 2. 推理测试
python例程不需要编译，可以直接运行，PCIe平台和SoC平台的测试参数和运行方式是相同的。

## 2.1 参数说明
resnet_opencv.py和resnet_bmcv.py的命令参数相同，以resnet_opencv.py的推理为例，参数说明如下：

```bash
usage:resnet_opencv.py [--input IMG_PATH] [--bmodel BMODEL] [--dev_id DEV_ID]
--input: 推理图片路径，可输入整个图片文件夹的路径；
--bmodel: 用于推理的bmodel路径，默认使用stage 0的网络进行推理；
--dev_id: 用于推理的tpu设备id。
```

### 2.2 测试图片
图片测试实例如下，支持对整个图片文件夹进行测试。
```bash
# 测试整个文件夹
python3 python/resnet_opencv.py --input datasets/imagenet_val_1k/img --bmodel models/BM1684X/resnet50_fp32_1b.bmodel --dev_id 0
```
测试结束后，会将预测结果保存在`results/resnet50_fp32_1b.bmodel_img_opencv_python_result.json`下，同时会打印预测结果、推理时间等信息。