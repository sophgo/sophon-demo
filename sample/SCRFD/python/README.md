[简体中文](./README.md) | [English](./README_EN.md)

# Python例程

## 目录

* [1. 环境准备](#1-环境准备)
    * [1.1 x86/arm PCIe平台](#11-x86arm-pcie平台)
    * [1.2 SoC平台](#12-soc平台)
* [2. 推理测试](#2-推理测试)
    * [2.1 参数说明](#21-参数说明)
    * [2.2 测试图片](#22-测试图片)
    * [2.3 测试视频](#23-测试视频)

python目录下提供了一系列Python例程，具体情况如下：

| 序号 |  Python例程      | 说明                                |
| ---- | ---------------- | -----------------------------------  |
| 1    | scrfd_opencv.py | 使用OpenCV解码、OpenCV前处理、SAIL推理 |
| 2    | scrfd_bmcv.py   | 使用SAIL解码、BMCV前处理、SAIL推理 |

## 1. 环境准备
### 1.1 x86/arm PCIe平台

如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon、sophon-opencv、sophon-ffmpeg和sophon-sail，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。

此外您还需要配置opencv等其他第三方库：
```bash
pip3 install opencv-python-headless
```

### 1.2 SoC平台

如果您使用SoC平台（如SE、SM系列边缘设备），并使用它测试本例程，刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包。您还需要交叉编译安装sophon-sail，具体可参考[交叉编译安装sophon-sail](../../../docs/Environment_Install_Guide.md#42-交叉编译安装sophon-sail)。

此外您还需要配置opencv等其他第三方库：
```bash
pip3 install opencv-python-headless
```
## 2. 推理测试
python例程不需要编译，可以直接运行，PCIe平台和SoC平台的测试参数和运行方式是相同的。
### 2.1 参数说明
scrfd_opencv.py和scrfd_bmcv.py的参数一致，以scrfd_opencv.py为例：
```bash
usage: scrfd_opencv.py [--input INPUT_PATH] [--bmodel BMODEL] [--dev_id DEV_ID]
                        [--conf_thresh CONF_THRESH] [--nms_thresh NMS_THRESH]
--input: 测试数据路径，可输入整个图片文件夹的路径或者视频路径；
--bmodel: 用于推理的bmodel路径，默认使用stage 0的网络进行推理；
--dev_id: 用于推理的tpu设备id；
--conf_thresh: 置信度阈值；
--nms_thresh: nms阈值；
--eval: 是否为精度评测使用,仅可在测试图片时使用，默认为False，        
        如果设置为False，则保存txt结果和图片结果，
        如果设置为True，则只保存txt结果用于精度评测。
```

### 2.2 测试图片
图片测试实例如下，支持对整个图片文件夹进行测试。
```bash
python3 python/scrfd_opencv.py --input datasets/test --bmodel models/BM1684/scrfd_10g_kps_fp32_1b.bmodel --dev_id 0 --conf_thresh 0.5 --nms_thresh 0.5 --eval True
```
测试结束后，当eval设置为True时，结果会自动保存在 `tools/prediction_dir` 文件夹下，不会保存预测的图片。
当eval设置为False时，会将预测的图片保存在`results/images`下，预测的结果保存在`results/txt_results/test`下，同时会打印预测结果、推理时间等信息。

> **注意**：  
> 1.bmcv例程暂时没有在图片上写字。

### 2.3 测试视频
视频测试实例如下，支持对视频流进行测试。
```bash
python3 python/scrfd_opencv.py --input datasets/face_det.mp4 --bmodel models/BM1684/scrfd_10g_kps_fp32_1b.bmodel --dev_id 0 --conf_thresh 0.5 --nms_thresh 0.5
```
测试结束后，会将预测的结果画在`results/face_det.avi`中，同时会打印预测结果、推理时间等信息。  
`scrfd_bmcv.py`不会保存视频，而是会将预测结果画在图片上并保存在`results/images`中。
