# P2PNet

## 目录

- [P2PNet](#p2pnet)
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
P2PNet是腾讯优图实验室提出的点对点网络（Point-to-Point Network，P2PNet），业界首创直接预测人头中心点的人群计数新范式，能够同时实现人群个体定位和人群计数，该算法在 2020 年 12 月份刷新 NWPU 榜单。本例程对[P2PNet官方开源仓库](https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet)的模型和算法进行移植，使之能在SOPHON BM1684和BM1684X上进行推理测试。

**数据集**: (https://www.datafountain.cn/datasets/5670)

## 2. 特性
* 支持BM1684X(x86 PCIe、SoC)和BM1684(x86 PCIe、SoC、arm PCIe)
* 支持FP32、FP16(BM1684X)、INT8模型编译和推理
* 支持基于BMCV预处理的C++推理
* 支持基于OpenCV和BMCV预处理的Python推理
* 支持单batch和多batch模型推理
* 支持图片和视频测试

## 3. 准备模型与数据
​本例程在`scripts`目录下提供了相关模型和数据的下载脚本`download.sh`，您也可以自己准备模型和数据集，并参考[4. 模型编译](#4-模型编译)进行模型转换。

```bash
# 安装unzip，若已安装请跳过，非ubuntu系统视情况使用yum或其他方式安装
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```

下载的模型包括：
```
./models
├── BM1684
│   ├── p2pnet_bm1684_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684的FP32 BModel，batch_size=1
│   ├── p2pnet_bm1684_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684的INT8 BModel，batch_size=1
│   └── p2pnet_bm1684_int8_4b.bmodel   # 使用TPU-MLIR编译，用于BM1684的INT8 BModel，batch_size=4
├── BM1684X
│   ├── p2pnet_bm1684x_fp32_1b.bmodel  # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=1
│   ├── p2pnet_bm1684x_fp16_1b.bmodel  # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=1
│   ├── p2pnet_bm1684x_int8_1b.bmodel  # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=1
│   └── p2pnet_bm1684x_int8_4b.bmodel  # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=4
└── onnx
    ├── p2pnet_1b.onnx                 # onnx模型，batch_size=1
    └── p2pnet_4b.onnx                 # onnx模型，batch_size=4
```
下载的数据包括：
```
./datasets
├── test                      # 测试数据集
│   ├──ground-truth           # 用于计算评价指标
│   └──images                 # 测试图片
├── calibration               # 用于模型量化
└── video
    └──video.avi              # 测试视频
```

## 4. 模型编译
导出的模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。建议使用TPU-MLIR编译BModel。

模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#2-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index/material/37/all.html)相应版本的SDK中获取)。

- 生成FP32 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（支持BM1684X），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684
#or
./scripts/gen_fp32bmodel_mlir.sh bm1684x
```

​执行上述命令会在`models/BM1684/`或`models/BM1684X`下生成`p2pnet_bm1684*_fp32_1b.bmodel`文件，即转换好的FP32 BModel。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X**），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x
```

​执行上述命令会在`models/BM1684X/`下生成`p2pnet_bm1684x_fp16_1b.bmodel`文件，即转换好的FP16 BModel。

- 生成INT8 BModel

​本例程在`scripts`目录下提供了量化INT8 BModel的脚本，请注意修改`gen_int8bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台（**支持BM1684/BM1684X**），如：

```shell
./scripts/gen_int8bmodel_mlir.sh bm1684
#或
./scripts/gen_int8bmodel_mlir.sh bm1684x
```

​上述脚本会在`models/BM1684`或`models/BM1684X`下生成`p2pnet_bm1684*_int8_1b.bmodel`等文件，即转换好的INT8 BModel。

## 5. 例程测试
- [C++例程](./cpp/README.md)
- [Python例程](./python/README.md)

## 6. 精度测试
### 6.1 测试方法

首先，参考[C++例程](cpp/README.md#69-测试数据集)或[Python例程](python/README.md#47-测试数据集)推理要测试的数据集，生成预测的txt文件。
然后，使用`tools`目录下的`eval_acc.py`脚本，将测试生成的txt文件与测试集标签mat文件进行对比，计算出准确率信息，命令如下：
```bash
# 请根据实际情况修改ground-truth和测试结果路径
python3 tools/eval_acc.py --gt_path datasets/test/ground-truth --result_path python/results/images
```
### 6.2 测试结果
根据本例程提供的数据集，测试结果如下：
|    测试平台    |         测试程序       |      	     测试模型         	  |   MAE  |  MSE  |
| ------------ | --------------------- | ------------------------------ | ------ | ----- |
| BM1684 pcie  | p2pnet_opencv.py      | p2pnet_bm1684_fp32_1b.bmodel   |  18.35 | 29.12 |
| BM1684 pcie  | p2pnet_opencv.py      | p2pnet_bm1684_int8_1b.bmodel   |  20.44 | 32.36 |
| BM1684 pcie  | p2pnet_bmcv.py        | p2pnet_bm1684_fp32_1b.bmodel   |  20.20 | 30.47 |
| BM1684 pcie  | p2pnet_bmcv.py        | p2pnet_bm1684_int8_1b.bmodel   |  20.66 | 32.96 |
| BM1684 pcie  | p2pnet_bmcv.pcie      | p2pnet_bm1684_fp32_1b.bmodel   |  18.15 | 28.69 |
| BM1684 pcie  | p2pnet_bmcv.pcie      | p2pnet_bm1684_int8_1b.bmodel   |  19.91 | 31.36 |
| BM1684X pcie | p2pnet_opencv.py      | p2pnet_bm1684x_fp32_1b.bmodel  |  18.35 | 29.12 |
| BM1684X pcie | p2pnet_opencv.py      | p2pnet_bm1684x_fp16_1b.bmodel  |  18.34 | 29.11 |
| BM1684X pcie | p2pnet_opencv.py      | p2pnet_bm1684x_int8_1b.bmodel  |  18.49 | 29.64 |
| BM1684X pcie | p2pnet_bmcv.py        | p2pnet_bm1684x_fp32_1b.bmodel  |  20.21 | 30.49 |
| BM1684X pcie | p2pnet_bmcv.py        | p2pnet_bm1684x_fp16_1b.bmodel  |  20.21 | 30.49 |
| BM1684X pcie | p2pnet_bmcv.py        | p2pnet_bm1684x_int8_1b.bmodel  |  20.34 | 30.71 |
| BM1684X pcie | p2pnet_bmcv.pcie      | p2pnet_bm1684x_fp32_1b.bmodel  |  18.06 | 28.48 |
| BM1684X pcie | p2pnet_bmcv.pcie      | p2pnet_bm1684x_fp16_1b.bmodel  |  18.09 | 28.51 |
| BM1684X pcie | p2pnet_bmcv.pcie      | p2pnet_bm1684x_int8_1b.bmodel  |  17.99 | 28.32 |

> **测试说明**：
> 1. batch_size=4和batch_size=1的模型精度一致；
> 2. SoC和PCIe的模型精度一致；

## 7. 性能测试
### 7.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684/p2pnet_bm1684_fp32_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

|                  测试模型                  | calculate time(ms) |
| ----------------------------------------- | -------------- |
| BM1684/p2pnet_bm1684_fp32_1b.bmodel  		  |     92.4       |
| BM1684/p2pnet_bm1684_int8_1b.bmodel  		  |     47.9       |
| BM1684/p2pnet_bm1684_int8_4b.bmodel  		  |			13.8       |
| BM1684X/p2pnet_bm1684x_fp32_1b.bmodel 	  |			158.5      |
| BM1684X/p2pnet_bm1684x_fp16_1b.bmodel 	  |			14.0       |
| BM1684X/p2pnet_bm1684x_int8_1b.bmodel 	  |			7.0        |
| BM1684X/p2pnet_bm1684x_int8_4b.bmodel 	  |			6.6        |

> **测试说明**：
> 1. 性能测试结果具有一定的波动性；
> 2. `calculate time`已折算为平均每张图片的推理时间；

### 7.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++例程打印的预处理时间、推理时间、后处理时间为整个batch处理的时间，需除以相应的batch size才是每张图片的处理时间。

在不同的测试平台上，使用不同的例程、模型测试`datasets/test/images`，性能测试结果如下：
|    测试平台  |     测试程序      |             测试模型        |decode_time|preprocess_time|inference_time|postprocess_time|
| ----------- | ---------------- | ----------------------------- | --------- | ------------- | ------------ | --------- |
| BM1684 pcie | p2pnet_opencv.py | p2pnet_bm1684_fp32_1b.bmodel  | 9.3       | 10.8          | 95.2         | 3.2       |
| BM1684 pcie | p2pnet_opencv.py | p2pnet_bm1684_int8_1b.bmodel  | 9.3       | 10.4          | 50.7         | 3.1       |
| BM1684 pcie | p2pnet_opencv.py | p2pnet_bm1684_int8_4b.bmodel  | 6.3       | 9.5           | 15.4         | 2.9       |
| BM1684 pcie | p2pnet_bmcv.py   | p2pnet_bm1684_fp32_1b.bmodel  | 4.6       | 2.8           | 93.1         | 3.2       |
| BM1684 pcie | p2pnet_bmcv.py   | p2pnet_bm1684_int8_1b.bmodel  | 4.6       | 2.8           | 48.7         | 3.2       |
| BM1684 pcie | p2pnet_bmcv.py   | p2pnet_bm1684_int8_4b.bmodel  | 4.6       | 3.2           | 14.1         | 2.9       |
| BM1684 pcie | p2pnet_bmcv.pcie | p2pnet_bm1684_fp32_1b.bmodel  | 4.0       | 1.0           | 92.4         | 0.8       |
| BM1684 pcie | p2pnet_bmcv.pcie | p2pnet_bm1684_int8_1b.bmodel  | 4.1       | 0.9           | 48.1         | 0.8       |
| BM1684 pcie | p2pnet_bmcv.pcie | p2pnet_bm1684_int8_4b.bmodel  | 4.1       | 0.9           | 13.8         | 0.7       |
| BM1684X pcie| p2pnet_opencv.py | p2pnet_bm1684x_fp32_1b.bmodel | 15.5      | 13.6          | 165.6        | 3.2       |
| BM1684X pcie| p2pnet_opencv.py | p2pnet_bm1684x_fp16_1b.bmodel | 15.5      | 13.6          | 21.0         | 3.2       |
| BM1684X pcie| p2pnet_opencv.py | p2pnet_bm1684x_int8_1b.bmodel | 15.5      | 13.4          | 13.9         | 3.1       |
| BM1684X pcie| p2pnet_opencv.py | p2pnet_bm1684x_int8_4b.bmodel | 8.2       | 9.6           | 12.7         | 3.0       |
| BM1684X pcie| p2pnet_bmcv.py   | p2pnet_bm1684x_fp32_1b.bmodel | 9.7       | 2.2           | 159.4        | 3.2       |
| BM1684X pcie| p2pnet_bmcv.py   | p2pnet_bm1684x_fp16_1b.bmodel | 9.7       | 2.2           | 15.0         | 3.1       |
| BM1684X pcie| p2pnet_bmcv.py   | p2pnet_bm1684x_int8_1b.bmodel | 9.7       | 2.2           | 7.9          | 3.1       |
| BM1684X pcie| p2pnet_bmcv.py   | p2pnet_bm1684x_int8_4b.bmodel | 9.7       | 2.2           | 7.3          | 3.0       |
| BM1684X pcie| p2pnet_bmcv.pcie | p2pnet_bm1684x_fp32_1b.bmodel | 5.0       | 1.0           | 158.5        | 1.1       |
| BM1684X pcie| p2pnet_bmcv.pcie | p2pnet_bm1684x_fp16_1b.bmodel | 5.0       | 1.0           | 14.0         | 1.1       |
| BM1684X pcie| p2pnet_bmcv.pcie | p2pnet_bm1684x_int8_1b.bmodel | 5.0       | 0.9           | 6.9          | 1.1       |
| BM1684X pcie| p2pnet_bmcv.pcie | p2pnet_bm1684x_int8_4b.bmodel | 5.0       | 0.9           | 6.6          | 1.1       |

> **测试说明**：
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. BM1684/1684X SoC的主控处理器均为8核 ARM A53 42320 DMIPS @2.3GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异，不同的阈值对后处理时间影响较大。

## 8. FAQ
bmcv目前不支持画点，本例程通过画框来实现，点数量很多时会出现 segmentation fault (core dumped)，不影响结果。

其他问题请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。