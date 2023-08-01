# Resnet

## 目录

* [1. 简介](#1-简介)
* [2. 特性](#2-特性)
* [3. 准备模型与数据](#3-准备模型与数据)
* [4. 模型编译](#4-模型编译)
* [5. 例程测试](#5-例程测试)
* [6. 精度测试](#6-精度测试)
  * [6.1 测试方法](#61-测试方法)
  * [6.2 测试结果](#62-测试结果)
* [7. 性能测试](#7-性能测试)
  * [7.1 bmrt_test](#71-bmrt_test)
  * [7.2 程序运行性能](#72-程序运行性能)
* [8. FAQ](#8-faq)
    
## 1. 简介
本例程对[torchvision Resnet](https://pytorch.org/vision/stable/models.html)的模型和算法进行移植，使之能在SOPHON BM1684和BM1684X上进行推理测试。

**论文:** [Resnet论文](https://arxiv.org/abs/1512.03385)

深度残差网络（Deep residual network, ResNet）是由于Kaiming He等在2015提出的深度神经网络结构，它利用残差学习来解决深度神经网络训练退化的问题。

在此非常感谢Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun等人的贡献。

## 2. 特性
* 支持BM1684X(x86 PCIe、SoC)和BM1684(x86 PCIe、SoC、arm PCIe)
* 支持FP32、FP16(BM1684X)、INT8模型编译和推理
* 支持基于OpenCV和BMCV预处理的C++推理
* 支持基于OpenCV和BMCV预处理的Python推理
* 支持单batch和多batch模型推理
* 支持图片测试

## 3. 准备模型与数据
建议使用TPU-MLIR编译BModel，Pytorch模型在编译前要导出成onnx模型。具体可参考[ResNet模型导出](./docs/ResNet_Export_Guide.md)。

同时，您需要准备用于测试的数据集，如果量化模型，还要准备用于量化的数据集。

本例程在`scripts`目录下提供了相关模型和数据集的下载脚本`download.sh`，您也可以自己准备模型和数据集，并参考[5. 模型编译](#5-模型编译)进行模型转换。
```bash
chmod +x ./scripts/*
./scripts/download.sh
```
执行后，模型保存至`models`，测试数据集下载并解压至`datasets/imagenet_val_1k`，量化数据集下载并解压至`datasets/cali_data`

下载的模型包括：
```
.
├── BM1684
│   ├── resnet50_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684的FP32 BModel，batch_size=1
│   ├── resnet50_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684的INT8 BModel，batch_size=1
│   └── resnet50_int8_4b.bmodel   # 使用TPU-MLIR编译，用于BM1684的INT8 BModel，batch_size=4
├── BM1684X
│   ├── resnet50_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=1
│   ├── resnet50_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=1
│   ├── resnet50_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=1
│   └── resnet50_int8_4b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=4
├── torch
│   ├── resnet50-11ad3fa6.pth                         # 原始模型
│   └── resnet50-11ad3fa6.torchscript.pt              # trace后的torchscript模型
└── onnx
    ├── resnet50_1b.onnx                               # 导出的onnx模型，batch_size=1
    ├── resnet50_4b.onnx                               # 导出的onnx模型，batch_size=4 
    └── resnet50_qtable                                # 量化效果不好时，可以使用该qtable设置敏感层
```

下载的数据包括：
```
./datasets
├── cali_data                   # 量化图片, 共200张   
│    
└── imagenet_val_1k                                      
    ├── img                     # 测试图片, 共1000张
    └── label.txt               # 标签文件 
```

## 4. 模型编译
导出的模型需要编译成BModel才能在SOPHON TPU上运行，建议使用TPU-MLIR编译BModel。

模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index/material/31/all.html)相应版本的SDK中获取)。

- 生成FP32 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684/BM1684X**），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684
#or
./scripts/gen_fp32bmodel_mlir.sh bm1684x
```

​执行上述命令会在`models/BM1684`或`models/BM1684X/`下生成`resnet50_fp32_1b.bmodel`文件，即转换好的FP32 BModel。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X**），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x
```

​执行上述命令会在`models/BM1684X/`下生成`resnet50_fp16_1b.bmodel`文件，即转换好的FP16 BModel。

- 生成INT8 BModel

​本例程在`scripts`目录下提供了量化INT8 BModel的脚本，请注意修改`gen_int8bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台（**支持BM1684/BM1684X**），如：

```shell
./scripts/gen_int8bmodel_mlir.sh bm1684
#或
./scripts/gen_int8bmodel_mlir.sh bm1684x
```

​上述脚本会在`models/BM1684`或`models/BM1684X/`下生成`resnet50_int8_1b.bmodel`等文件，即转换好的INT8 BModel。

## 5. 例程测试
* [C++例程](cpp/README.md)
* [Python例程](python/README.md)

## 6. 精度测试
### 6.1 测试方法

首先，参考[C++例程](cpp/README.md#32-测试图片)或[Python例程](python/README.md#22-测试图片)推理要测试的数据集，生成预测的json文件，注意修改相关参数。  
然后，使用`tools`目录下的`eval_imagenet.py`脚本，将预测结果文件与测试集标签文件进行对比，计算出分类准确率。具体的测试命令如下：
```bash
# 请根据实际情况修改文件路径
python3 tools/eval_imagenet.py --gt_path datasets/imagenet_val_1k/label.txt --result_json cpp/resnet_opencv/results/resnet50_fp32_1b.bmodel_img_opencv_cpp_result.json
```
### 6.2 测试结果
在imagenet_val_1k数据集上，精度测试结果如下：
|   测试平台    |      测试程序       |        测试模型        | ACC(%) |
| ------------ | ----------------   | ---------------------- | ------ |
| BM1684 PCIe  | resnet_opencv.py   | resnet50_fp32_1b.bmodel  | 80.10  |
| BM1684 PCIe  | resnet_opencv.py   | resnet50_int8_1b.bmodel  | 78.70  |
| BM1684 PCIe  | resnet_bmcv.py     | resnet50_fp32_1b.bmodel  | 79.90  |
| BM1684 PCIe  | resnet_bmcv.py     | resnet50_int8_1b.bmodel  | 78.50  |
| BM1684 PCIe  | resnet_opencv.pcie | resnet50_fp32_1b.bmodel  | 80.20  |
| BM1684 PCIe  | resnet_opencv.pcie | resnet50_int8_1b.bmodel  | 78.20  |
| BM1684 PCIe  | resnet_bmcv.pcie   | resnet50_fp32_1b.bmodel  | 79.90  |
| BM1684 PCIe  | resnet_bmcv.pcie   | resnet50_int8_1b.bmodel  | 78.50  |
| BM1684X PCIe | resnet_opencv.py   | resnet50_fp32_1b.bmodel  | 80.10  |
| BM1684X PCIe | resnet_opencv.py   | resnet50_fp16_1b.bmodel  | 80.10  |
| BM1684X PCIe | resnet_opencv.py   | resnet50_int8_1b.bmodel  | 79.10  |
| BM1684X PCIe | resnet_bmcv.py     | resnet50_fp32_1b.bmodel  | 80.00  |
| BM1684X PCIe | resnet_bmcv.py     | resnet50_fp16_1b.bmodel  | 80.00  |
| BM1684X PCIe | resnet_bmcv.py     | resnet50_int8_1b.bmodel  | 79.40  |
| BM1684X PCIe | resnet_opencv.pcie | resnet50_fp32_1b.bmodel  | 80.00  |
| BM1684X PCIe | resnet_opencv.pcie | resnet50_fp16_1b.bmodel  | 80.00  |
| BM1684X PCIe | resnet_opencv.pcie | resnet50_int8_1b.bmodel  | 79.20  |
| BM1684X PCIe | resnet_bmcv.pcie   | resnet50_fp32_1b.bmodel  | 80.00  |
| BM1684X PCIe | resnet_bmcv.pcie   | resnet50_fp16_1b.bmodel  | 80.00  |
| BM1684X PCIe | resnet_bmcv.pcie   | resnet50_int8_1b.bmodel  | 79.40  |

> **测试说明**：  
1. batch_size=4和batch_size=1的模型精度一致；
2. SoC和PCIe的模型准确率一致；

## 7. 性能测试
### 7.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684/resnet50_fp32_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

|                  测试模型      | calculate time(ms) |
| ----------------------------- | ----------------- |
| BM1684/resnet50_fp32_1b.bmodel  | 6.35              |
| BM1684/resnet50_int8_1b.bmodel  | 3.92              |
| BM1684/resnet50_int8_4b.bmodel  | 1.14              |
| BM1684X/resnet50_fp32_1b.bmodel | 8.84              |
| BM1684X/resnet50_fp16_1b.bmodel | 1.57              |
| BM1684X/resnet50_int8_1b.bmodel | 1.07              |
| BM1684X/resnet50_int8_4b.bmodel | 0.79              |

> **测试说明**：  
1. 性能测试结果具有一定的波动性；
2. `calculate time`已折算为平均每张图片的推理时间；
3. SoC和PCIe的测试结果基本一致。

### 7.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++例程打印的预处理时间、推理时间、后处理时间为整个batch处理的时间，需除以相应的batch size才是每张图片的处理时间。

在不同的测试平台上，使用不同的例程、模型测试`datasets/imagenet_val_1k`，性能测试结果如下：
|    测试平台  |     测试程序        |        测试模型       |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ------------------ | --------------------- | -------- | --------- | --------- | --------- |
| BM1684 SoC  | resnet_opencv.py   | resnet50_fp32_1b.bmodel | 10.28    | 8.06      | 9.03      | 0.31      |
| BM1684 SoC  | resnet_opencv.py   | resnet50_int8_1b.bmodel | 10.19    | 7.95      | 5.91      | 0.33      |
| BM1684 SoC  | resnet_opencv.py   | resnet50_int8_4b.bmodel | 10.06    | 8.00      | 3.24      | 0.11      |
| BM1684 SoC  | resnet_bmcv.py     | resnet50_fp32_1b.bmodel | 1.34     | 1.52      | 6.90      | 0.25      |
| BM1684 SoC  | resnet_bmcv.py     | resnet50_int8_1b.bmodel | 1.35     | 1.52      | 4.05      | 0.24      |
| BM1684 SoC  | resnet_bmcv.py     | resnet50_int8_4b.bmodel | 1.19     | 1.43      | 1.24      | 0.10      |
| BM1684 SoC  | resnet_opencv.soc  | resnet50_fp32_1b.bmodel | 1.47     | 6.23      | 6.49      | 0.14      |
| BM1684 SoC  | resnet_opencv.soc  | resnet50_int8_1b.bmodel | 1.47     | 6.27      | 3.64      | 0.15      |
| BM1684 SoC  | resnet_opencv.soc  | resnet50_int8_4b.bmodel | 1.29     | 6.26      | 1.11      | 0.12      |
| BM1684 SoC  | resnet_bmcv.soc    | resnet50_fp32_1b.bmodel | 3.90     | 2.45      | 6.49      | 0.11      |
| BM1684 SoC  | resnet_bmcv.soc    | resnet50_int8_1b.bmodel | 2.91     | 2.45      | 3.63      | 0.13      |
| BM1684 SoC  | resnet_bmcv.soc    | resnet50_int8_4b.bmodel | 2.85     | 2.41      | 1.11      | 0.11      |
| BM1684X SoC | resnet_opencv.py   | resnet50_fp32_1b.bmodel | 4.03     | 8.57      | 11.41     | 0.30      |
| BM1684X SoC | resnet_opencv.py   | resnet50_fp16_1b.bmodel | 1.85     | 8.49      | 4.29      | 0.29      |
| BM1684X SoC | resnet_opencv.py   | resnet50_int8_1b.bmodel | 1.83     | 8.33      | 3.61      | 0.29      |
| BM1684X SoC | resnet_opencv.py   | resnet50_int8_4b.bmodel | 1.75     | 8.47      | 3.04      | 0.10      |
| BM1684X SoC | resnet_bmcv.py     | resnet50_fp32_1b.bmodel | 1.25     | 0.73      | 9.20      | 0.26      |
| BM1684X SoC | resnet_bmcv.py     | resnet50_fp16_1b.bmodel | 1.26     | 0.73      | 2.04      | 0.26      |
| BM1684X SoC | resnet_bmcv.py     | resnet50_int8_1b.bmodel | 1.31     | 0.77      | 1.56      | 0.26      |
| BM1684X SoC | resnet_bmcv.py     | resnet50_int8_4b.bmodel | 1.02     | 0.62      | 0.90      | 0.10      |
| BM1684X SoC | resnet_bmcv.soc    | resnet50_fp32_1b.bmodel | 2.55     | 0.47      | 8.72      | 0.12      |
| BM1684X SoC | resnet_bmcv.soc    | resnet50_fp16_1b.bmodel | 2.85     | 0.47      | 1.58      | 0.12      |
| BM1684X SoC | resnet_bmcv.soc    | resnet50_int8_1b.bmodel | 2.55     | 0.45      | 1.06      | 0.11      |
| BM1684X SoC | resnet_bmcv.soc    | resnet50_int8_4b.bmodel | 2.49     | 0.42      | 0.79      | 0.09      |
| BM1684X SoC | resnet_opencv.soc  | resnet50_fp32_1b.bmodel | 1.17     | 5.97      | 8.59      | 0.14      |
| BM1684X SoC | resnet_opencv.soc  | resnet50_fp16_1b.bmodel | 1.25     | 5.98      | 1.56      | 0.15      |
| BM1684X SoC | resnet_opencv.soc  | resnet50_int8_1b.bmodel | 1.15     | 5.92      | 1.04      | 0.15      |
| BM1684X SoC | resnet_opencv.soc  | resnet50_int8_4b.bmodel | 0.96     | 5.95      | 0.78      | 0.11      |


> **测试说明**：  
1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
2. 性能测试结果具有一定的波动性，建议多次测试取平均值；

## 8. FAQ
请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。
