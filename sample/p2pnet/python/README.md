# Python例程
## 目录

- [1. 环境准备](#1-环境准备)
  - [1.1 x86/arm PCIE模式](#11-x86/arm PCIE模式)
  - [1.2 soc模式](#12-soc模式)
- [2. 推理测试](#2-推理测试)
  - [2.1 参数说明](#21-参数说明)
  - [2.2 测试图像](#22-测试图像)
  - [2.3 测试视频](#22-测试视频)

python目录下提供了一系列Python例程，具体情况如下：

| 序号 |  Python例程      | 说明                                |
| ---- | ---------------- | -----------------------------------  |
| 1    | p2pnet_opencv.py | 使用OpenCV解码、OpenCV前处理、SAIL推理 、OpenCV后处理|
| 2    | p2pnet_bmcv.py   | 使用SAIL解码、BMCV前处理、SAIL推理、OpenCV后处理 |
| 3    | p2pnet_onnx.py   | 使用OpenCV解码、OpenCV前处理、ONNX推理、OpenCV后处理 |
| 4    | p2pnet_trace_pt.py | 使用OpenCV解码、OpenCV前处理、PyTorch推理、OpenCV后处理 |

## 1. 环境准备

​支持以下环境运行本程序。

### 1.1 x86/arm PCIE模式

如果您在x86/arm平台安装了PCIe加速卡，并使用它测试本例程，您需要安装libsophon、sophon-opencv、sophon-ffmpeg和sophon-sail,具体请参考[x86-pcie平台的开发和运行环境搭建](../../docs/Environment_Install_Guide.md#2-x86-pcie平台的开发和运行环境搭建)。
此外您可能还需要安装其他第三方库：
```bash
pip3 install 'opencv-python-headless<4.3'
```
运行p2pnet_onnx.py需要安装:
```bash
pip3 install 'onnx'
pip3 install 'onnxruntime'
```
运行p2pnet_trace_pt.py需要安装pytoch、torchvision，如果用GPU，还需要安装cudakit。


### 1.2 SOC模式

如果您使用SoC平台测试本例程，您需要交叉编译安装sophon-sail，具体可参考[交叉编译安装sophon-sail](../../docs/Environment_Install_Guide.md#32-交叉编译安装sophon-sail)。
此外您可能还需要安装其他第三方库：
```bash
pip3 install 'opencv-python-headless<4.3'
```

## 2.  推理测试
python例程不需要编译，可以直接运行，PCIe平台和SoC平台的测试参数和运行方式是相同的。SoC模式下不具备pytorch、onnx环境，不建议使用p2pnet_trace_pt.py、p2pnet_onnx.py测试

### 2.1 参数说明

python程序默认有一套参数，其中p2pnet_opencv.py和p2pnet_bmcv.py参数格式相同，p2pnet_onnx.py和p2pnet_trace_pt.py参数格式相同，请注意根据实际情况进行传参，下面分别以p2pnet_opencv.py和p2pnet_onnx.py为例进行详细说明：
p2pnet_opencv.py：
```bash
usage: p2pnet_opencv.py [-h] [--input INPUT] [--bmodel BMODEL]
                        [--dev_id DEV_ID]

optional arguments:
  -h, --help       show this help message and exit
  --input INPUT    input image path
  --bmodel BMODEL  bmodel path
  --dev_id DEV_ID  device id
```
p2pnet_onnx.py：
```bash
usage: p2pnet_onnx.py [-h] [--model MODEL] [--batch_size BATCH_SIZE]
                      [--input INPUT]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         onnx model path
  --batch_size BATCH_SIZE
                        batch size
  --input INPUT         input image path
```
### 2.2 测试图像
图片测试实例如下，支持对整个图片文件夹进行测试。
```bash
python3 python/p2pnet_opencv.py --input datasets/ShanghaiTech/ShanghaiTech-Dataset/ShanghaiTech/part_A/test_data/images --bmodel models/BM1684/p2pnet_bm1684_fp32_1b.bmodel --dev_id 0
```
测试结束后，预测的图像和文本文件保存在`results`目录下。

### 2.3 测试视频
测试实例如下，支持对视频进行测试。
```bash
python3 python/p2pnet_opencv.py --input datasets/video/video.avi --bmodel models/BM1684/p2pnet_bm1684_fp32_1b.bmodel --dev_id 0
```
测试结束后，预测的图像和文本文件保存在`results`目录下。
