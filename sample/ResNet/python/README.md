# Python例程

## 目录

* [1. 环境准备](#1-环境准备)
    * [1.1 x86 PCIe平台](#11-x86-pcie平台)
* [2. 推理测试](#2-推理测试)
    * [2.1 参数说明](#21-参数说明)
    * [2.2 测试图片](#21-测试图片)

python目录下提供了一系列Python例程，具体情况如下：

| 序号   | Python例程      | 说明                                |
| ---- | ---------------- | -----------------------------------  |
| 1    | resnet_opencv.py | 使用OpenCV解码、OpenCV前处理、SAIL推理 |

## 1. 环境准备
### 1.1 x86 PCIe平台

目前仅支持在x86 PCIe平台测试本例程。除了安装tpuv7-driver和tpuv7-runtime之外，此外您可能还需要安装其他第三方库：
```bash
pip3 install opencv-python-headless
```
## 2. 推理测试
python例程不需要编译，可以直接运行。

## 2.1 参数说明
以resnet_opencv.py的推理为例，参数说明如下：

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
python3 python/resnet_opencv.py --input datasets/imagenet_val_1k/img --bmodel models/BM1690/resnet50_int8_1b.bmodel --dev_id 0
```
测试结束后，会将预测结果保存在`results/resnet50_fp32_1b.bmodel_img_opencv_python_result.json`下，同时会打印预测结果、推理时间等信息。