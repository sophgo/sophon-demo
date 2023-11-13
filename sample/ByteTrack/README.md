# ByteTrack

## 目录

- [ByteTrack](#bytetrack)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
  - [3. 准备模型与数据](#3-准备模型与数据)
  - [4. 模型编译](#4-模型编译)
  - [5. 例程测试](#5-例程测试)
  - [6. 精度测试](#6-精度测试)
    - [6.1 测试方法](#61-测试方法)
    - [6.2 测试结果](#62-测试结果)
  - [7. 性能测试](#7-性能测试)
    - [7.1 bmrt\_test](#71-bmrt_test)
    - [7.2 程序运行性能](#72-程序运行性能)
  - [8. FAQ](#8-faq)

## 1. 简介
ByteTrack是一个简单、快速、强大的多目标跟踪器，且不依赖特征提取模型。

**论文** (https://arxiv.org/abs/2110.06864)

**源代码** (https://github.com/ifzhang/ByteTrack)

## 2. 特性
* 支持BM1684X(x86 PCIe、SoC)和BM1684(x86 PCIe、SoC、arm PCIe)
* 支持检测模块和跟踪模块解藕，可适配各种检测器，本例程主要以YOLOv5作为检测器
* 支持基于BMCV预处理的C++推理
* 支持基于OpenCV预处理的Python推理
* 支持MOT格式数据集(即图片文件夹)和单视频测试

## 3. 准备模型与数据
本例程使用YOLOv5的目标检测模型，详情请参考[YOLOv5](../YOLOv5/README.md#3-准备模型与数据)。

​在`scripts`目录下提供了相关模型和数据的下载脚本`download.sh`。

```bash
# 安装unzip，若已安装请跳过，非ubuntu系统视情况使用yum或其他方式安装
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```

下载的模型包括：
```bash
./models
├── BM1684
│   ├── yolov5s_v6.1_3output_fp32_1b.bmodel   # 从YOLOv5例程中获取，用于BM1684的FP32 BModel，batch_size=1
│   ├── yolov5s_v6.1_3output_int8_1b.bmodel   # 从YOLOv5例程中获取，用于BM1684的INT8 BModel，batch_size=1
│   └── yolov5s_v6.1_3output_int8_4b.bmodel   # 从YOLOv5例程中获取，用于BM1684的INT8 BModel，batch_size=4
├── BM1684X
│   ├── yolov5s_v6.1_3output_fp32_1b.bmodel   # 从YOLOv5例程中获取，用于BM1684X的FP32 BModel，batch_size=1
│   ├── yolov5s_v6.1_3output_fp16_1b.bmodel   # 从YOLOv5例程中获取，用于BM1684X的FP16 BModel，batch_size=1
│   ├── yolov5s_v6.1_3output_int8_1b.bmodel   # 从YOLOv5例程中获取，用于BM1684X的INT8 BModel，batch_size=1
└──  └── yolov5s_v6.1_3output_int8_4b.bmodel  # 从YOLOv5例程中获取，用于BM1684X的INT8 BModel，batch_size=4
```
下载的数据包括：
```
./datasets
├── test_car_person_1080P.mp4                 # 测试视频
└── mot15_trainset                            # MOT15的训练集，这里用于评价指标测试。
```

## 4. 模型编译

本例程使用YOLOv5的目标检测模型，详情请参考[YOLOv5](../YOLOv5/README.md#4-模型编译)。

## 5. 例程测试
- [C++例程](./cpp/README.md)
- [Python例程](./python/README.md)

## 6. 精度测试
### 6.1 测试方法

首先，参考[C++例程](cpp/README.md#32-测试MOT数据集)或[Python例程](python/README.md#22-测试MOT数据集)推理要测试的数据集，生成包含目标追踪结果的txt文件，注意修改数据集`datasets/mot15_trainset/ADL-Rundle-6/img1`。
然后，使用`tools`目录下的`eval_mot15.py`脚本，将测试生成的txt文件与测试集标签txt文件进行对比，计算出目标追踪的一系列评价指标，在BM1684x SoC上运行命令：
```bash
# 安装motmetrics，若已安装请跳过
pip3 install motmetrics
# 请根据实际情况修改程序路径和txt文件路径
python3 tools/eval_mot15.py --gt_file datasets/mot15_trainset/ADL-Rundle-6/gt/gt.txt --ts_file python/results/mot_eval/ADL-Rundle-6_yolov5s_v6.1_3output_int8_1b.bmodel.txt
```
运行结果：
```bash
MOTA = 0.5260531044120582
     num_frames      IDF1       IDP       IDR      Rcll      Prcn    GT  MT  PT  ML   FP    FN  IDsw  FM      MOTA      MOTP
acc         525  0.602846  0.733543  0.511679  0.614893  0.881511  5009  10  12   2  414  1929    31  60  0.526053  0.212196
```
### 6.2 测试结果
这里使用数据集ADL-Rundle-6，记录MOTA作为精度指标，精度测试结果如下：
| 测试平台     | 测试程序              | 测试模型                            | MOTA  |
| ------------ | --------------------- | ----------------------------------- | ----- |
| BM1684 PCIe  | bytetrack_opencv.py   | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.516 |
| BM1684 PCIe  | bytetrack_opencv.py   | yolov5s_v6.1_3output_int8_1b.bmodel | 0.514 |
| BM1684 PCIe  | bytetrack_opencv.pcie | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.512 |
| BM1684 PCIe  | bytetrack_opencv.pcie | yolov5s_v6.1_3output_int8_1b.bmodel | 0.504 |
| BM1684 PCIe  | bytetrack_eigen.pcie  | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.512 |
| BM1684 PCIe  | bytetrack_eigen.pcie  | yolov5s_v6.1_3output_int8_1b.bmodel | 0.504 |
| BM1684X PCIe | bytetrack_opencv.py   | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.516 |
| BM1684X PCIe | bytetrack_opencv.py   | yolov5s_v6.1_3output_fp16_1b.bmodel | 0.516 |
| BM1684X PCIe | bytetrack_opencv.py   | yolov5s_v6.1_3output_int8_1b.bmodel | 0.526 |
| BM1684X PCIe | bytetrack_opencv.pcie | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.508 |
| BM1684X PCIe | bytetrack_opencv.pcie | yolov5s_v6.1_3output_fp16_1b.bmodel | 0.508 |
| BM1684X PCIe | bytetrack_opencv.pcie | yolov5s_v6.1_3output_int8_1b.bmodel | 0.548 |
| BM1684X PCIe | bytetrack_eigen.pcie  | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.508 |
| BM1684X PCIe | bytetrack_eigen.pcie  | yolov5s_v6.1_3output_fp16_1b.bmodel | 0.508 |
| BM1684X PCIe | bytetrack_eigen.pcie  | yolov5s_v6.1_3output_int8_1b.bmodel | 0.548 |


> **测试说明**：
1. batch_size=4和batch_size=1的模型精度一致；
2. SoC和PCIe的模型精度一致；


## 7. 性能测试
### 7.1 bmrt_test

本例程使用YOLOv5的目标检测模型，详情请参考[YOLOv5](../YOLOv5/README.md#71-bmrt_test)。

### 7.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理、tracker更新时间。C++例程打印的预处理时间、推理时间、后处理为整个batch处理的时间，需除以相应的batch size才是每张图片的处理时间。这里**只统计平均每帧的track_time**，解码、目标检测模型的时间请参考[YOLOV5](../YOLOv5/README.md#72-程序运行性能)

在不同的测试平台上，使用不同的例程、模型测试`datasets/mot15_trainset/ADL-Rundle-6/img1`，性能测试结果如下：
| 测试平台     | 测试程序              | 测试模型                            | track_time |
| ------------ | --------------------- | ----------------------------------- | ---------- |
| BM1684 PCIe  | bytetrack_opencv.py   | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.81       |
| BM1684 PCIe  | bytetrack_opencv.py   | yolov5s_v6.1_3output_int8_1b.bmodel | 0.66       |
| BM1684 PCIe  | bytetrack_opencv.py   | yolov5s_v6.1_3output_int8_4b.bmodel | 0.60       |
| BM1684 PCIe  | bytetrack_opencv.pcie | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.80       |
| BM1684 PCIe  | bytetrack_opencv.pcie | yolov5s_v6.1_3output_int8_1b.bmodel | 0.61       |
| BM1684 PCIe  | bytetrack_opencv.pcie | yolov5s_v6.1_3output_int8_4b.bmodel | 0.54       |
| BM1684 PCIe  | bytetrack_eigen.pcie  | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.22       |
| BM1684 PCIe  | bytetrack_eigen.pcie  | yolov5s_v6.1_3output_int8_1b.bmodel | 0.17       |
| BM1684 PCIe  | bytetrack_eigen.pcie  | yolov5s_v6.1_3output_int8_4b.bmodel | 0.13       |
| BM1684x PCIe | bytetrack_opencv.py   | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.73       |
| BM1684x PCIe | bytetrack_opencv.py   | yolov5s_v6.1_3output_fp16_1b.bmodel | 0.71       |
| BM1684x PCIe | bytetrack_opencv.py   | yolov5s_v6.1_3output_int8_1b.bmodel | 0.70       |
| BM1684x PCIe | bytetrack_opencv.py   | yolov5s_v6.1_3output_int8_4b.bmodel | 0.63       |
| BM1684X PCIe | bytetrack_opencv.pcie | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.78       |
| BM1684X PCIe | bytetrack_opencv.pcie | yolov5s_v6.1_3output_fp16_1b.bmodel | 0.65       |
| BM1684X PCIe | bytetrack_opencv.pcie | yolov5s_v6.1_3output_int8_1b.bmodel | 0.58       |
| BM1684X PCIe | bytetrack_opencv.pcie | yolov5s_v6.1_3output_int8_4b.bmodel | 0.62       |
| BM1684X PCIe | bytetrack_eigen.pcie  | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.21       |
| BM1684X PCIe | bytetrack_eigen.pcie  | yolov5s_v6.1_3output_fp16_1b.bmodel | 0.18       |
| BM1684X PCIe | bytetrack_eigen.pcie  | yolov5s_v6.1_3output_int8_1b.bmodel | 0.16       |
| BM1684X PCIe | bytetrack_eigen.pcie  | yolov5s_v6.1_3output_int8_4b.bmodel | 0.15       |

> **测试说明**：
1. 时间单位均为毫秒(ms)；
2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
3. BM1684/1684X SoC的处理器均为8核 ARM A53 42320 DMIPS @2.3GHz，PCIe上的性能由于处理器的不同可能存在较大差异；

## 8. FAQ
请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。