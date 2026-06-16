# MobileNetV4

## 目录

- [MobileNetV4](#mobilenetv4)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
    - [2.1 目录结构说明](#21-目录结构说明)
    - [2.2 SDK特性](#22-sdk特性)
  - [3. 数据准备与模型编译](#3-数据准备与模型编译)
    - [3.1 数据准备](#31-数据准备)
    - [3.2 模型编译](#32-模型编译)
  - [4. 例程测试](#4-例程测试)
  - [5. 精度测试](#5-精度测试)
    - [5.1 测试方法](#51-测试方法)
    - [5.2 测试结果](#52-测试结果)
  - [6. 性能测试](#6-性能测试)
    - [6.1 bmrt\_test](#61-bmrt_test)
    - [6.2 程序运行性能](#62-程序运行性能)
  - [8. FAQ](#8-faq)

## 1. 简介
MobileNetV4例程对[timm MobileNetV4](https://huggingface.co/timm/mobilenetv4_conv_medium.e250_r384_in1k)模型和算法进行移植，支持在SOPHON BM1684X/BM1688/CV186X上进行推理测试。

**论文:** [MobileNetV4 - Universal Models for the Mobile Ecosystem](https://arxiv.org/abs/2404.10518)

MobileNetV4 是 Google 在 2024 年提出的移动端通用模型架构，通过引入 Universal Inverted Bottleneck (UIB) 搜索空间和 Mobile MQA 注意力机制，在移动端实现高效的图像分类。

在此非常感谢 Danfeng Qin, Chas Leichner 等人的贡献。

## 2. 特性

### 2.1 目录结构说明
```bash
├── cpp                   # 存放C++例程及其README
|   ├──README.md
|   ├──mobilenetv4_bmcv   # C++例程
├── docs                  # 存放本例程专用文档，如ONNX导出、移植常见问题等
├── pics                  # 存放README等说明文档中用到的图片
├── python                # 存放Python例程及其README
|   ├──README.md
|   ├──mobilenetv4_bmcv.py     # Python例程
|   └──...                # Python例程共用功能的封装。
├── README.md             # 本例程的中文指南
├── scripts               # 存放模型编译、数据下载、自动测试等shell脚本
└── tools                 # 存放精度测试、性能比对等python脚本
```

### 2.2 SDK特性
* 支持BM1688(SoC)、CV186X(SoC)、BM1684X(x86 PCIe、SoC、riscv PCIe)
* 支持FP32、FP16(BM1684X/BM1688/CV186X)、INT8模型编译和推理
* 支持C++、Python推理
* 支持图片测试

## 3. 数据准备与模型编译

### 3.1 数据准备

本例程在`scripts`目录下提供了相关模型和数据的下载脚本`download.sh`，**如果您希望自己准备模型和数据集，可以跳过本小节，参考[3.2 模型编译](#32-模型编译)进行模型转换。**

```bash
chmod -R +x scripts/
./scripts/download.sh
```

执行后，模型保存至`models`，测试数据集下载并解压至`datasets/imagenet_val_1k`，量化数据集下载并解压至`datasets/cali_data`

下载的模型包括：
```bash
models/
├── BM1684X # 在BM1684X上运行的模型
│   ├── mobilenetv4_conv_medium_fp32_1b.bmodel   # 使用TPU-MLIR编译，FP32 BModel，batch_size=1
│   ├── mobilenetv4_conv_medium_fp16_1b.bmodel   # 使用TPU-MLIR编译，FP16 BModel，batch_size=1
│   ├── mobilenetv4_conv_medium_int8_1b.bmodel   # 使用TPU-MLIR编译，INT8 BModel，batch_size=1
│   └── mobilenetv4_conv_medium_int8_4b.bmodel   # 使用TPU-MLIR编译，INT8 BModel，batch_size=4
├── BM1688 # 在BM1688上运行的模型
│   ├── mobilenetv4_conv_medium_fp32_1b.bmodel       # FP32 BModel，batch_size=1
│   ├── mobilenetv4_conv_medium_fp16_1b.bmodel       # FP16 BModel，batch_size=1
│   ├── mobilenetv4_conv_medium_int8_1b.bmodel       # INT8 BModel，batch_size=1
│   ├── mobilenetv4_conv_medium_int8_4b.bmodel       # INT8 BModel，batch_size=4
│   ├── mobilenetv4_conv_medium_fp32_1b_2core.bmodel # FP32 BModel，batch_size=1, num_core=2
│   ├── mobilenetv4_conv_medium_fp16_1b_2core.bmodel # FP16 BModel，batch_size=1, num_core=2
│   ├── mobilenetv4_conv_medium_int8_1b_2core.bmodel # INT8 BModel，batch_size=1, num_core=2
│   └── mobilenetv4_conv_medium_int8_4b_2core.bmodel # INT8 BModel，batch_size=4, num_core=2
├── CV186X # 在CV186X上运行的模型
│   ├── mobilenetv4_conv_medium_fp32_1b.bmodel   # FP32 BModel，batch_size=1
│   ├── mobilenetv4_conv_medium_fp16_1b.bmodel   # FP16 BModel，batch_size=1
│   ├── mobilenetv4_conv_medium_int8_1b.bmodel   # INT8 BModel，batch_size=1
│   └── mobilenetv4_conv_medium_int8_4b.bmodel   # INT8 BModel，batch_size=4
├── torch
│   ├── mobilenetv4_conv_medium.pth                # 原始模型
│   └── mobilenetv4_conv_medium.torchscript.pt     # trace后的torchscript模型
└── onnx
    └── mobilenetv4_conv_medium.onnx               # 导出的onnx模型
```

下载的数据包括：
```bash
./datasets
├── cali_data                                      # 量化数据集
└── imagenet_val_1k                                # 测试数据集
    ├── img                     # 测试图片, 共1000张
    └── label.txt               # 标签文件
```

### 3.2 模型编译

**如果您不编译模型，只想直接使用下载的数据集和模型，可以跳过本小节。**

源模型需要编译成BModel才能在SOPHON TPU上运行，源模型在编译前要导出成onnx模型，具体可参考[模型导出脚本](./scripts/export_onnx.py)。同时，您需要准备用于测试的数据集，如果量化模型，还要准备用于量化的数据集。

建议使用TPU-MLIR编译BModel，模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录，并使用本例程提供的脚本将onnx模型编译为BModel。脚本中命令的详细说明可参考《TPU-MLIR开发手册》(请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的SDK中获取)。

- 生成FP32 BModel

本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684x #bm1688/cv186x
```

执行上述命令会在`models/BM1684X`等文件夹下生成转换好的FP32 BModel。

- 生成FP16 BModel

本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x #bm1688/cv186x
```

执行上述命令会在`models/BM1684X/`等文件夹下生成转换好的FP16 BModel。

- 生成INT8 BModel

本例程在`scripts`目录下提供了量化INT8 BModel的脚本，请注意修改`gen_int8bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台（**支持BM1684X/BM1688/CV186X**），如：

```shell
./scripts/gen_int8bmodel_mlir.sh bm1684x #bm1688/cv186x
```

上述脚本会在`models/BM1684X`等文件夹下生成转换好的INT8 BModel。

## 4. 例程测试
- [C++例程](./cpp/README.md)
- [Python例程](./python/README.md)

## 5. 精度测试
### 5.1 测试方法

首先，参考[C++例程](cpp/README.md#32-测试图片)或[Python例程](python/README.md#22-测试图片)推理要测试的数据集，生成预测的json文件，注意修改相关参数。
然后，使用`tools`目录下的`eval_imagenet.py`脚本，将预测结果文件与测试集标签文件进行对比，计算出分类准确率。具体的测试命令如下：
```bash
# 请根据实际情况修改文件路径
python3 tools/eval_imagenet.py --gt_path datasets/imagenet_val_1k/label.txt --result_json cpp/mobilenetv4_bmcv/results/mobilenetv4_conv_medium_fp32_1b.bmodel_img_bmcv_cpp_result.json
```

### 5.2 测试结果
在imagenet_val_1k数据集上，精度测试结果如下：
|   测试平台   |      测试程序           |        测试模型                            | ACC(%) |
| ------------ | ---------------------- | ---------------------------------------- | ------ |
|   SE7-32    |  mobilenetv4_bmcv.py   |      mobilenetv4_conv_medium_fp32_1b.bmodel       |  78.50 |
|   SE7-32    |  mobilenetv4_bmcv.py   |      mobilenetv4_conv_medium_fp16_1b.bmodel       |  78.50 |
|   SE7-32    |  mobilenetv4_bmcv.py   |      mobilenetv4_conv_medium_int8_1b.bmodel       |  77.30 |
|   SE7-32    |  mobilenetv4_bmcv.py   |      mobilenetv4_conv_medium_int8_4b.bmodel       |  77.30 |
|   SE7-32    |  mobilenetv4_bmcv.soc  |      mobilenetv4_conv_medium_fp32_1b.bmodel       |  78.50 |
|   SE7-32    |  mobilenetv4_bmcv.soc  |      mobilenetv4_conv_medium_fp16_1b.bmodel       |  78.50 |
|   SE7-32    |  mobilenetv4_bmcv.soc  |      mobilenetv4_conv_medium_int8_1b.bmodel       |  77.30 |
|   SE7-32    |  mobilenetv4_bmcv.soc  |      mobilenetv4_conv_medium_int8_4b.bmodel       |  77.30 |
|   SE9-16    |  mobilenetv4_bmcv.py   |      mobilenetv4_conv_medium_fp32_1b.bmodel       |  79.20 |
|   SE9-16    |  mobilenetv4_bmcv.py   |      mobilenetv4_conv_medium_fp16_1b.bmodel       |  79.20 |
|   SE9-16    |  mobilenetv4_bmcv.py   |      mobilenetv4_conv_medium_int8_1b.bmodel       |  78.50 |
|   SE9-16    |  mobilenetv4_bmcv.py   |      mobilenetv4_conv_medium_int8_4b.bmodel       |  78.50 |
|   SE9-16    |  mobilenetv4_bmcv.soc  |      mobilenetv4_conv_medium_fp32_1b.bmodel       |  79.20 |
|   SE9-16    |  mobilenetv4_bmcv.soc  |      mobilenetv4_conv_medium_fp16_1b.bmodel       |  79.20 |
|   SE9-16    |  mobilenetv4_bmcv.soc  |      mobilenetv4_conv_medium_int8_1b.bmodel       |  78.50 |
|   SE9-16    |  mobilenetv4_bmcv.soc  |      mobilenetv4_conv_medium_int8_4b.bmodel       |  78.50 |
|   SE9-16    |  mobilenetv4_bmcv.soc  |      mobilenetv4_conv_medium_int8_4b_2core.bmodel |  78.50 |
|    SE9-8    |  mobilenetv4_bmcv.py   |      mobilenetv4_conv_medium_fp32_1b.bmodel       |  79.20 |
|    SE9-8    |  mobilenetv4_bmcv.py   |      mobilenetv4_conv_medium_fp16_1b.bmodel       |  79.20 |
|    SE9-8    |  mobilenetv4_bmcv.py   |      mobilenetv4_conv_medium_int8_1b.bmodel       |  78.50 |
|    SE9-8    |  mobilenetv4_bmcv.py   |      mobilenetv4_conv_medium_int8_4b.bmodel       |  78.50 |
|    SE9-8    |  mobilenetv4_bmcv.soc  |      mobilenetv4_conv_medium_fp32_1b.bmodel       |  79.20 |
|    SE9-8    |  mobilenetv4_bmcv.soc  |      mobilenetv4_conv_medium_fp16_1b.bmodel       |  79.20 |
|    SE9-8    |  mobilenetv4_bmcv.soc  |      mobilenetv4_conv_medium_int8_1b.bmodel       |  78.50 |
|    SE9-8    |  mobilenetv4_bmcv.soc  |      mobilenetv4_conv_medium_int8_4b.bmodel       |  78.50 |

> **测试说明**：
> 1. 由于sdk版本之间可能存在差异，实际运行结果与本表有<0.01的精度误差是正常的；
> 2. 在搭载了相同TPU和SOPHONSDK的PCIe或SoC平台上，相同程序的精度一致，SE5系列对应BM1684，SE7系列对应BM1684X，SE9系列中，SE9-16对应BM1688，SE9-8对应CV186X；

## 6. 性能测试
### 6.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684X/mobilenetv4_conv_medium_fp32_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

|    测试平台  |              测试模型           | calculate time(ms) |
| ----------- | ------------------------------ | ----------------- |
|   SE7-32    | BM1684X/mobilenetv4_conv_medium_fp32_1b.bmodel |           1.90  |
|   SE7-32    | BM1684X/mobilenetv4_conv_medium_fp16_1b.bmodel |           0.98  |
|   SE7-32    | BM1684X/mobilenetv4_conv_medium_int8_1b.bmodel |           0.57  |
|   SE7-32    | BM1684X/mobilenetv4_conv_medium_int8_4b.bmodel |           0.27  |
|   SE9-16    | BM1688/mobilenetv4_conv_medium_fp32_1b.bmodel  |           9.50  |
|   SE9-16    | BM1688/mobilenetv4_conv_medium_fp16_1b.bmodel  |           1.81  |
|   SE9-16    | BM1688/mobilenetv4_conv_medium_int8_1b.bmodel  |           0.82  |
|   SE9-16    | BM1688/mobilenetv4_conv_medium_int8_4b.bmodel  |           0.51  |
|   SE9-16    | BM1688/mobilenetv4_conv_medium_int8_4b_2core.bmodel |      0.38  |
|    SE9-8    | CV186X/mobilenetv4_conv_medium_fp32_1b.bmodel  |           9.49  |
|    SE9-8    | CV186X/mobilenetv4_conv_medium_fp16_1b.bmodel  |           1.95  |
|    SE9-8    | CV186X/mobilenetv4_conv_medium_int8_1b.bmodel  |           0.97  |
|    SE9-8    | CV186X/mobilenetv4_conv_medium_int8_4b.bmodel  |           0.56  |

> **测试说明**：
> 1. 性能测试结果具有一定的波动性；
> 2. `calculate time`已折算为平均每张图片的推理时间；
> 3. SoC和PCIe的测试结果基本一致。

### 6.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++和Python例程打印的时间已经折算为单张图片的处理时间。

在不同的测试平台上，使用不同的例程、模型测试`datasets/imagenet_val_1k`，性能测试结果如下：
|    测试平台  |     测试程序      |        测试模型        |decode_time|preprocess_time|inference_time|postprocess_time|
| ----------- | ---------------- | ---------------------- | -------- | --------- | --------- | --------- |
|   SE7-32    |  mobilenetv4_bmcv.py   |      mobilenetv4_conv_medium_fp32_1b.bmodel       |      1.68       |      0.55       |      4.20       |      0.28       |
|   SE7-32    |  mobilenetv4_bmcv.py   |      mobilenetv4_conv_medium_fp16_1b.bmodel       |      1.69       |      0.55       |      3.00       |      0.30       |
|   SE7-32    |  mobilenetv4_bmcv.py   |      mobilenetv4_conv_medium_int8_1b.bmodel       |      1.45       |      0.49       |      2.15       |      0.25       |
|   SE7-32    |  mobilenetv4_bmcv.py   |      mobilenetv4_conv_medium_int8_4b.bmodel       |      1.48       |      0.48       |      0.45       |      0.11       |
|   SE7-32    |  mobilenetv4_bmcv.soc  |      mobilenetv4_conv_medium_fp32_1b.bmodel       |      1.28       |      0.61       |      3.36       |      0.14       |
|   SE7-32    |  mobilenetv4_bmcv.soc  |      mobilenetv4_conv_medium_fp16_1b.bmodel       |      1.15       |      0.57       |      1.47       |      0.13       |
|   SE7-32    |  mobilenetv4_bmcv.soc  |      mobilenetv4_conv_medium_int8_1b.bmodel       |      1.44       |      0.63       |      1.50       |      0.20       |
|   SE7-32    |  mobilenetv4_bmcv.soc  |      mobilenetv4_conv_medium_int8_4b.bmodel       |      1.12       |      0.55       |      0.87       |      0.13       |
|   SE9-16    |  mobilenetv4_bmcv.py   |      mobilenetv4_conv_medium_fp32_1b.bmodel       |      2.64       |      1.06       |      9.90       |      0.36       |
|   SE9-16    |  mobilenetv4_bmcv.py   |      mobilenetv4_conv_medium_fp16_1b.bmodel       |      2.52       |      1.05       |      2.20       |      0.35       |
|   SE9-16    |  mobilenetv4_bmcv.py   |      mobilenetv4_conv_medium_int8_1b.bmodel       |      2.50       |      1.06       |      1.25       |      0.35       |
|   SE9-16    |  mobilenetv4_bmcv.py   |      mobilenetv4_conv_medium_int8_4b.bmodel       |      2.21       |      0.90       |      0.62       |      0.14       |
|   SE9-16    |  mobilenetv4_bmcv.soc  |      mobilenetv4_conv_medium_fp32_1b.bmodel       |      2.43       |      1.25       |      9.46       |      0.21       |
|   SE9-16    |  mobilenetv4_bmcv.soc  |      mobilenetv4_conv_medium_fp16_1b.bmodel       |      2.32       |      1.25       |      1.80       |      0.21       |
|   SE9-16    |  mobilenetv4_bmcv.soc  |      mobilenetv4_conv_medium_int8_1b.bmodel       |      2.12       |      1.25       |      0.82       |      0.21       |
|   SE9-16    |  mobilenetv4_bmcv.soc  |      mobilenetv4_conv_medium_int8_4b.bmodel       |      2.33       |      1.11       |      0.52       |      0.17       |
|   SE9-16    |  mobilenetv4_bmcv.soc  |      mobilenetv4_conv_medium_int8_4b_2core.bmodel |      2.01       |      1.11       |      0.36       |      0.17       |
|    SE9-8    |  mobilenetv4_bmcv.py   |      mobilenetv4_conv_medium_fp32_1b.bmodel       |      2.78       |      1.05       |      9.73       |      0.34       |
|    SE9-8    |  mobilenetv4_bmcv.py   |      mobilenetv4_conv_medium_fp16_1b.bmodel       |      2.45       |      1.05       |      2.17       |      0.33       |
|    SE9-8    |  mobilenetv4_bmcv.py   |      mobilenetv4_conv_medium_int8_1b.bmodel       |      2.41       |      1.04       |      1.15       |      0.32       |
|    SE9-8    |  mobilenetv4_bmcv.py   |      mobilenetv4_conv_medium_int8_4b.bmodel       |      2.13       |      0.91       |      0.61       |      0.13       |
|    SE9-8    |  mobilenetv4_bmcv.soc  |      mobilenetv4_conv_medium_fp32_1b.bmodel       |      2.11       |      1.25       |      9.34       |      0.20       |
|    SE9-8    |  mobilenetv4_bmcv.soc  |      mobilenetv4_conv_medium_fp16_1b.bmodel       |      2.55       |      1.25       |      1.78       |      0.21       |
|    SE9-8    |  mobilenetv4_bmcv.soc  |      mobilenetv4_conv_medium_int8_1b.bmodel       |      2.05       |      1.24       |      0.79       |      0.20       |
|    SE9-8    |  mobilenetv4_bmcv.soc  |      mobilenetv4_conv_medium_int8_4b.bmodel       |      2.23       |      1.14       |      0.51       |      0.17       |

> **测试说明**：
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. SE5-16/SE7-32的主控处理器均为8核CA53@2.3GHz，SE9-16为8核CA53@1.6GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异。

## 8. FAQ
请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。
