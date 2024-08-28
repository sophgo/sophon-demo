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
* 支持检测模块和跟踪模块解耦，可适配各种检测器，本例程以YOLOv5作为检测器
* 支持基于BMCV预处理的C++推理
* 支持基于OpenCV预处理的Python推理
* 支持MOT格式数据集(即图片文件夹)和单视频测试

## 3. 准备模型与数据
本例程使用YOLOv5的目标检测模型，详情请参考[YOLOv5](../YOLOv5/README.md#3-数据准备与模型编译)。

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
│   └── yolov5s_v6.1_3output_int8_4b.bmodel   # 从YOLOv5例程中获取，用于BM1684X的INT8 BModel，batch_size=4
│── BM1688
│   ├── yolov5s_v6.1_3output_fp16_1b.bmodel       # 从YOLOv5例程中获取，用于BM1688的FP16 BModel，batch_size=1, num_core=1
│   ├── yolov5s_v6.1_3output_fp32_1b.bmodel       # 从YOLOv5例程中获取，用于BM1688的FP32 BModel，batch_size=1, num_core=1
│   ├── yolov5s_v6.1_3output_int8_1b.bmodel       # 从YOLOv5例程中获取，用于BM1688的INT8 BModel，batch_size=1, num_core=1
│   ├── yolov5s_v6.1_3output_int8_4b.bmodel       # 从YOLOv5例程中获取，用于BM1688的INT8 BModel，batch_size=1, num_core=1
│   ├── yolov5s_v6.1_3output_fp16_1b_2core.bmodel # 从YOLOv5例程中获取，用于BM1688的FP16 BModel，batch_size=1, num_core=2
│   ├── yolov5s_v6.1_3output_fp32_1b_2core.bmodel # 从YOLOv5例程中获取，用于BM1688的FP32 BModel，batch_size=1, num_core=2
│   ├── yolov5s_v6.1_3output_int8_1b_2core.bmodel # 从YOLOv5例程中获取，用于BM1688的INT8 BModel，batch_size=1, num_core=2
│   └── yolov5s_v6.1_3output_int8_4b_2core.bmodel # 从YOLOv5例程中获取，用于BM1688的INT8 BModel，batch_size=4, num_core=2
└── CV186X
    ├── yolov5s_v6.1_3output_fp16_1b.bmodel   # 从YOLOv5例程中获取，用于CV186X的FP32 BModel，batch_size=1
    ├── yolov5s_v6.1_3output_fp32_1b.bmodel   # 从YOLOv5例程中获取，用于CV186X的FP16 BModel，batch_size=1
    ├── yolov5s_v6.1_3output_int8_1b.bmodel   # 从YOLOv5例程中获取，用于CV186X的INT8 BModel，batch_size=1
    └── yolov5s_v6.1_3output_int8_4b.bmodel   # 从YOLOv5例程中获取，用于CV186X的INT8 BModel，batch_size=4

```
下载的数据包括：
```bash
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
| SE5-16       | bytetrack_opencv.py   | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.516 |
| SE5-16       | bytetrack_opencv.py   | yolov5s_v6.1_3output_int8_1b.bmodel | 0.514 |
| SE5-16       | bytetrack_opencv.soc  | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.510 |
| SE5-16       | bytetrack_opencv.soc  | yolov5s_v6.1_3output_int8_1b.bmodel | 0.507 |
| SE5-16       | bytetrack_eigen.soc   | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.510 |
| SE5-16       | bytetrack_eigen.soc   | yolov5s_v6.1_3output_int8_1b.bmodel | 0.507 |
| SE7-32       | bytetrack_opencv.py   | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.516 |
| SE7-32       | bytetrack_opencv.py   | yolov5s_v6.1_3output_fp16_1b.bmodel | 0.516 |
| SE7-32       | bytetrack_opencv.py   | yolov5s_v6.1_3output_int8_1b.bmodel | 0.526 |
| SE7-32       | bytetrack_opencv.soc  | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.525 |
| SE7-32       | bytetrack_opencv.soc  | yolov5s_v6.1_3output_fp16_1b.bmodel | 0.525 |
| SE7-32       | bytetrack_opencv.soc  | yolov5s_v6.1_3output_int8_1b.bmodel | 0.538 |
| SE7-32       | bytetrack_eigen.soc   | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.525 |
| SE7-32       | bytetrack_eigen.soc   | yolov5s_v6.1_3output_fp16_1b.bmodel | 0.525 |
| SE7-32       | bytetrack_eigen.soc   | yolov5s_v6.1_3output_int8_1b.bmodel | 0.538 |
| SE9-16       | bytetrack_opencv.py   | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.516 |
| SE9-16       | bytetrack_opencv.py   | yolov5s_v6.1_3output_fp16_1b.bmodel | 0.516 |
| SE9-16       | bytetrack_opencv.py   | yolov5s_v6.1_3output_int8_1b.bmodel | 0.501 |
| SE9-16       | bytetrack_opencv.soc  | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.524 |
| SE9-16       | bytetrack_opencv.soc  | yolov5s_v6.1_3output_fp16_1b.bmodel | 0.524 |
| SE9-16       | bytetrack_opencv.soc  | yolov5s_v6.1_3output_int8_1b.bmodel | 0.491 |
| SE9-16       | bytetrack_eigen.soc   | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.524 |
| SE9-16       | bytetrack_eigen.soc   | yolov5s_v6.1_3output_fp16_1b.bmodel | 0.524 |
| SE9-16       | bytetrack_eigen.soc   | yolov5s_v6.1_3output_int8_1b.bmodel | 0.491 |
| SE9-8       | bytetrack_opencv.py   | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.516 |
| SE9-8       | bytetrack_opencv.py   | yolov5s_v6.1_3output_fp16_1b.bmodel | 0.516 |
| SE9-8       | bytetrack_opencv.py   | yolov5s_v6.1_3output_int8_1b.bmodel | 0.501 |
| SE9-8       | bytetrack_opencv.soc  | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.524 |
| SE9-8       | bytetrack_opencv.soc  | yolov5s_v6.1_3output_fp16_1b.bmodel | 0.524 |
| SE9-8       | bytetrack_opencv.soc  | yolov5s_v6.1_3output_int8_1b.bmodel | 0.491 |
| SE9-8       | bytetrack_eigen.soc   | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.524 |
| SE9-8       | bytetrack_eigen.soc   | yolov5s_v6.1_3output_fp16_1b.bmodel | 0.524 |
| SE9-8       | bytetrack_eigen.soc   | yolov5s_v6.1_3output_int8_1b.bmodel | 0.491 |

> **测试说明**：
> 1. batch_size=4和batch_size=1的模型精度一致；
> 2. 由于sdk版本之间可能存在差异，实际运行结果与本表有<0.01的精度误差是正常的；
> 3. 在搭载了相同TPU和SOPHONSDK的PCIe或SoC平台上，相同程序的精度一致，SE5系列对应BM1684，SE7系列对应BM1684X，SE9-16对应BM1688，SE9-8对应CV186X；
> 4. BM1688 num_core=2的模型与num_core=1的模型精度基本一致；


## 7. 性能测试
### 7.1 bmrt_test

本例程使用YOLOv5的目标检测模型，详情请参考[YOLOv5](../YOLOv5/README.md#61-bmrt_test)。

### 7.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理、tracker更新时间。C++和Python例程打印的时间已经折算为单张图片的处理时间。这里**只统计平均每帧的track_time**，解码、目标检测模型的时间请参考[YOLOV5](../YOLOv5/README.md#62-程序运行性能)

在不同的测试平台上，使用不同的例程、模型测试`datasets/mot15_trainset/ADL-Rundle-6/img1`，性能测试结果如下：
| 测试平台     | 测试程序              | 测试模型                            | track_time |
| ------------ | --------------------- | ----------------------------------- | ---------- |
|   SE5-16    |   bytetrack_opencv.py   |yolov5s_v6.1_3output_fp32_1b.bmodel|      5.77       |
|   SE5-16    |   bytetrack_opencv.py   |yolov5s_v6.1_3output_int8_1b.bmodel|      5.33       |
|   SE5-16    |   bytetrack_opencv.py   |yolov5s_v6.1_3output_int8_4b.bmodel|      5.49       |
|   SE5-16    |   bytetrack_opencv.soc  |yolov5s_v6.1_3output_fp32_1b.bmodel|      0.60       |
|   SE5-16    |   bytetrack_opencv.soc  |yolov5s_v6.1_3output_int8_1b.bmodel|      0.53       |
|   SE5-16    |   bytetrack_opencv.soc  |yolov5s_v6.1_3output_int8_4b.bmodel|      0.51       |
|   SE5-16    |   bytetrack_eigen.soc   |yolov5s_v6.1_3output_fp32_1b.bmodel|      0.35       |
|   SE5-16    |   bytetrack_eigen.soc   |yolov5s_v6.1_3output_int8_1b.bmodel|      0.31       |
|   SE5-16    |   bytetrack_eigen.soc   |yolov5s_v6.1_3output_int8_4b.bmodel|      0.30       |
|   SE7-32    |   bytetrack_opencv.py   |yolov5s_v6.1_3output_fp32_1b.bmodel|      5.81       |
|   SE7-32    |   bytetrack_opencv.py   |yolov5s_v6.1_3output_fp16_1b.bmodel|      5.81       |
|   SE7-32    |   bytetrack_opencv.py   |yolov5s_v6.1_3output_int8_1b.bmodel|      5.79       |
|   SE7-32    |   bytetrack_opencv.py   |yolov5s_v6.1_3output_int8_4b.bmodel|      5.99       |
|   SE7-32    |   bytetrack_opencv.soc  |yolov5s_v6.1_3output_fp32_1b.bmodel|      0.59       |
|   SE7-32    |   bytetrack_opencv.soc  |yolov5s_v6.1_3output_fp16_1b.bmodel|      0.59       |
|   SE7-32    |   bytetrack_opencv.soc  |yolov5s_v6.1_3output_int8_1b.bmodel|      0.61       |
|   SE7-32    |   bytetrack_opencv.soc  |yolov5s_v6.1_3output_int8_4b.bmodel|      0.58       |
|   SE7-32    |    bytetrack_eigen.soc  |yolov5s_v6.1_3output_fp32_1b.bmodel|      0.35       |
|   SE7-32    |    bytetrack_eigen.soc  |yolov5s_v6.1_3output_fp16_1b.bmodel|      0.35       |
|   SE7-32    |    bytetrack_eigen.soc  |yolov5s_v6.1_3output_int8_1b.bmodel|      0.36       |
|   SE7-32    |    bytetrack_eigen.soc  |yolov5s_v6.1_3output_int8_4b.bmodel|      0.34       |
|   SE9-16    |   bytetrack_opencv.py   |yolov5s_v6.1_3output_fp32_1b.bmodel|      8.21       |
|   SE9-16    |   bytetrack_opencv.py   |yolov5s_v6.1_3output_fp16_1b.bmodel|      8.18       |
|   SE9-16    |   bytetrack_opencv.py   |yolov5s_v6.1_3output_int8_1b.bmodel|      7.94       |
|   SE9-16    |   bytetrack_opencv.py   |yolov5s_v6.1_3output_int8_4b.bmodel|      8.24       |
|   SE9-16    |   bytetrack_opencv.soc  |yolov5s_v6.1_3output_fp32_1b.bmodel|      2.04       |
|   SE9-16    |   bytetrack_opencv.soc  |yolov5s_v6.1_3output_fp16_1b.bmodel|      2.03       |
|   SE9-16    |   bytetrack_opencv.soc  |yolov5s_v6.1_3output_int8_1b.bmodel|      1.93       |
|   SE9-16    |   bytetrack_opencv.soc  |yolov5s_v6.1_3output_int8_4b.bmodel|      1.92       |
|   SE9-16    |   bytetrack_eigen.soc   |yolov5s_v6.1_3output_fp32_1b.bmodel|      0.49       |
|   SE9-16    |   bytetrack_eigen.soc   |yolov5s_v6.1_3output_fp16_1b.bmodel|      0.49       |
|   SE9-16    |   bytetrack_eigen.soc   |yolov5s_v6.1_3output_int8_1b.bmodel|      0.47       |
|   SE9-16    |   bytetrack_eigen.soc   |yolov5s_v6.1_3output_int8_4b.bmodel|      0.46       |
|   SE9-16    |   bytetrack_opencv.py   |yolov5s_v6.1_3output_fp32_1b_2core.bmodel|      8.24       |
|   SE9-16    |   bytetrack_opencv.py   |yolov5s_v6.1_3output_fp16_1b_2core.bmodel|      8.23       |
|   SE9-16    |   bytetrack_opencv.py   |yolov5s_v6.1_3output_int8_1b_2core.bmodel|      7.92       |
|   SE9-16    |   bytetrack_opencv.py   |yolov5s_v6.1_3output_int8_4b_2core.bmodel|      8.18       |
|   SE9-16    |   bytetrack_opencv.soc  |yolov5s_v6.1_3output_fp32_1b_2core.bmodel|      2.04       |
|   SE9-16    |   bytetrack_opencv.soc  |yolov5s_v6.1_3output_fp16_1b_2core.bmodel|      2.03       |
|   SE9-16    |   bytetrack_opencv.soc  |yolov5s_v6.1_3output_int8_1b_2core.bmodel|      1.93       |
|   SE9-16    |   bytetrack_opencv.soc  |yolov5s_v6.1_3output_int8_4b_2core.bmodel|      1.92       |
|   SE9-16    |   bytetrack_eigen.soc   |yolov5s_v6.1_3output_fp32_1b_2core.bmodel|      0.49       |
|   SE9-16    |   bytetrack_eigen.soc   |yolov5s_v6.1_3output_fp16_1b_2core.bmodel|      0.49       |
|   SE9-16    |   bytetrack_eigen.soc   |yolov5s_v6.1_3output_int8_1b_2core.bmodel|      0.47       |
|   SE9-16    |   bytetrack_eigen.soc   |yolov5s_v6.1_3output_int8_4b_2core.bmodel|      0.46       |
|    SE9-8    |   bytetrack_opencv.py   |yolov5s_v6.1_3output_fp32_1b.bmodel|      8.49       |
|    SE9-8    |   bytetrack_opencv.py   |yolov5s_v6.1_3output_fp16_1b.bmodel|      8.96       |
|    SE9-8    |   bytetrack_opencv.py   |yolov5s_v6.1_3output_int8_1b.bmodel|      8.04       |
|    SE9-8    |   bytetrack_opencv.py   |yolov5s_v6.1_3output_int8_4b.bmodel|      8.93       |
|    SE9-8    |  bytetrack_opencv.soc   |yolov5s_v6.1_3output_fp32_1b.bmodel|      0.83       |
|    SE9-8    |  bytetrack_opencv.soc   |yolov5s_v6.1_3output_fp16_1b.bmodel|      0.80       |
|    SE9-8    |  bytetrack_opencv.soc   |yolov5s_v6.1_3output_int8_1b.bmodel|      0.74       |
|    SE9-8    |  bytetrack_opencv.soc   |yolov5s_v6.1_3output_int8_4b.bmodel|      0.74       |
|    SE9-8    |   bytetrack_eigen.soc   |yolov5s_v6.1_3output_fp32_1b.bmodel|      0.51       |
|    SE9-8    |   bytetrack_eigen.soc   |yolov5s_v6.1_3output_fp16_1b.bmodel|      0.48       |
|    SE9-8    |   bytetrack_eigen.soc   |yolov5s_v6.1_3output_int8_1b.bmodel|      0.46       |
|    SE9-8    |   bytetrack_eigen.soc   |yolov5s_v6.1_3output_int8_4b.bmodel|      0.47       |

> **测试说明**：  
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. SE5-16/SE7-32的主控处理器均为8核CA53@2.3GHz，SE9-16的主控处理器为8核CA53@1.6GHz，SE9-8为6核CA53@1.6GHz,PCIe上的性能由于处理器的不同可能存在较大差异；
> 4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异，不同的阈值对后处理时间影响较大。 


## 8. FAQ
请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。