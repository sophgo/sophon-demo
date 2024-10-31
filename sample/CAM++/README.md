# CAM++

## 目录

- [CAM++](#campplus)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
  - [3. 准备模型和链接库](#3-准备模型和链接库)
    - [3.1 使用提供的模型](#31-使用提供的模型)
    - [3.2 自行编译模型](#32-自行编译模型)
  - [4. 例程测试](#4-例程测试)
  - [5. 程序性能测试](#5-程序性能测试)
    - [5.1. bmrt_test](#51-bmrt_test)
    - [5.2. 程序运行性能](#51-程序运行性能)
  - [6. FAQ](#6-FAQ)

## 1. 简介

CAM++模型是基于密集连接时延神经网络的说话人识别模型。相比于一些主流的说话人识别模型，比如ResNet34和ECAPA-TDNN，CAM++具有更准确的说话人识别性能和更快的推理速度。该模型可以用于说话人确认、说话人日志、语音合成、说话人风格转化等多项任务。关于该模型的其他特性，请前往源repo查看：[iic/speech_campplus_sv_zh-cn_16k-common](https://www.modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common)。本例程对[3D-Speaker](https://github.com/modelscope/3D-Speaker)中的`iic/speech_campplus_sv_zh-cn_16k-common`模型进行移植，使之能在SOPHON BM1684X/BM1688/CV186X上进行推理测试。

该例程支持在V23.09LTS SP3及以上的BM1684X SOPHONSDK, 或在v1.7.0及以上的BM1688/CV186X SOPHONSDK上运行，支持在插有1684X加速卡(SC7系列)的x86主机上运行，也可以在1684X SoC设备（如SE7、SM7、Airbox等）上运行，也支持在BM1688/CV186X Soc设备（如SE9-16）上运行。

在SoC上运行需要额外进行环境配置，请参照[运行环境准备](#3-运行环境准备)完成环境部署。

## 2.特性

* 支持BM1684X(x86 PCIe、SoC)，BM1688(SoC)，CV186X(Soc)
* 支持FP32(BM1684X/BM1688/CV186X)模型编译和推理
* 支持基于BMRT的C++例程

## 3. 准备模型和链接库

该模型目前支持在BM1684X，BM1688和CV186X上运行，已提供编译好的bmodel。

### 3.1 使用提供的模型

​本例程在`scripts`目录下提供了下载脚本`download.sh`

```bash
# 安装unzip，若已安装请跳过，非ubuntu系统视情况使用yum或其他方式安装
sudo apt-get update
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```

执行程序后，当前目录下的文件如下：

```shell
.
├── cpp
│   ├── CMakeLists.txt
│   ├── main.cpp                                    #主程序
│   ├── campplus.hpp                                #主要函数头文件
│   ├── dependencies                                #编译所需头文件目录
│   └── README.md                                   #C++例程说明
├── python
│   ├── campplus.py                                 #主程序
│   └── README.md                                   #Python例程说明
├── docs
│   └── Campplus_Export_Guide.md                    #模型导出及编译指南
├── models
│   ├── BM1684X                                     #使用TPU-MLIR编译，用于BM1684X的模型
│   │   ├── campplus_bm1684x_fp32_1b.bmodel
│   ├── CV186X                                      #使用TPU-MLIR编译，用于CV186X的模型
│   │   ├── campplus_cv186x_fp32_1b.bmodel
│   └── BM1688                                      #使用TPU-MLIR编译，用于BM1688的模型
│       └── campplus_bm1688_fp32_1b.bmodel
├── README.md                                       #例程说明
├── scripts                                         #下载及模型编译脚本等
│   ├── gen_fp32bmodel_mlir.sh                      #编译bmodel的脚本
│   └── download.sh                                 #下载脚本
└── tools
    └── campplus_npy.py                             #生成数据集的程序
```

### 3.2 自行编译模型
此部分请参考[Campplus模型导出与编译](./docs/Campplus_Export_Guide.md)

## 4. 例程测试
C++例程的详细编译请参考[C++例程](./cpp/README.md)
python例程的详细说明请参考[Python例程](./python/README.md)

## 5. 程序性能测试
### 5.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684X/campplus_bm1684x_fp32_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是理论推理时间。
测试各个模型的理论推理时间，结果如下：

| 测试模型                                       | calculate time(ms) |
| ---------------------------------------------- | ------------------ |
| BM1684X/campplus_bm1684x_fp32_1b.bmodel        |        58.38       |
| BM1688/campplus_bm1688_fp32_1b.bmodel          |       101.50       |
| CV186X/campplus_cv186x_fp32_1b.bmodel          |       100.69       |

> **测试说明**：
1. 性能测试结果具有一定的波动性；
2. `calculate time`已折算为每个输入的平均推理时间；
3. SoC和PCIe的测试结果基本一致。
4. 例程为动态模型，相比于静态模型会有一定性能损失

### 5.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的音频解码时间、预处理时间、推理时间、后处理时间。C++和Python例程打印的时间已经折算为单个语音文件的处理时间。

在不同的测试平台上，使用不同的例程、模型测试`datasets/test`，性能测试结果如下：
|   测试平台  |    测试程序   |            测试模型           |decode_time|preprocess_time|inference_time|postprocess_time|
| ----------- | ------------- | ----------------------------- | --------- | ------------- | ------------ | -------------- |
|   SE7-32    |campplus.py    |campplus_bm1684x_fp32_1b.bmodel|   27.49   |      13.60    |      56.59   |      0.00      |
|   SE7-32    |campplus       |campplus_bm1684x_fp32_1b.bmodel|    0.18   |      78.68    |      55.28   |      0.02      |
|   SE9-16    |campplus.py    |campplus_bm1684x_fp32_1b.bmodel|   40.06   |      17.88    |      99.79   |      0.00      |
|   SE9-16    |campplus       |campplus_bm1684x_fp32_1b.bmodel|    0.26   |     109.53    |      99.55   |      0.02      |
|   SE9-8     |campplus.py    |campplus_bm1684x_fp32_1b.bmodel|   40.08   |      22.56    |      99.79   |      0.00      |
|   SE9-8     |campplus       |campplus_bm1684x_fp32_1b.bmodel|    0.28   |     109.64    |      99.35   |      0.02      |

> **测试说明**：
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每个音频文件处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. SE7-32的主控处理器均为8核CA53@2.3GHz，SE9-16的主控处理器为8核CA53@1.6GHz，SE9-8为6核CA53@1.6GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 4. 不同的测试音频可能存在较大差异；
> 5. Campplus的后处理只有d2s（内存搬运），基本可以忽略。
> 6. python程序由于import相关库的时间以及相关函数初始化的时间未统计，因此前处理的时长较C++版本短很多。

## 6. FAQ
请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。
