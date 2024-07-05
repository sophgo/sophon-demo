[简体中文](./README.md) | [English](./README_EN.md)

# SAM

## 目录

- [SAM](#sam)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
  - [3. 准备模型与数据](#3-准备模型与数据)
  - [4. 模型编译](#4-模型编译)
  - [5. 例程测试](#5-例程测试)
    - [5.1 Python例程](#51-python例程)
    - [5.2 web\_ui例程](#52-web_ui例程)
  - [6. 性能测试](#6-性能测试)
    - [6.1 bmrt\_test](#61-bmrt_test)
    - [6.2 程序运行性能](#62-程序运行性能)
  - [7. FAQ](#7-faq)
  
## 1. 简介
​SAM是Meta提出的一个分割一切的提示型模型，其在1100万张图像上训练了超过10亿个掩码，实现了强大的零样本泛化，突破了分割界限。本例程对[​SAM官方开源仓库](https://github.com/facebookresearch/segment-anything)的模型和算法进行移植，使之能在SOPHON BM1684X上进行推理测试。

## 2. 特性
* 支持BM1684X(x86 PCIe、SoC)
* 图像压缩(embedding)部分支持FP16 1batch(BM1684X)模型编译和推理
* 图像推理(mask_decoder)部分支持FP32 1batch、FP16 1batch(BM1684X)模型编译和推理
* 支持基于OpenCV的Python推理
* 支持单点和box输入的模型推理，并输出最高置信度mask或置信度前三的mask
* 支持图片测试
* 支持无需点框输入的自动图掩码生成

**注意：
本repo将图像压缩（embedding）和图像推理（mask_decoder）分为两个bmodel运行；
图像推理部分最后一层resize未编入bmodel模型；详情请参考'scripts/gen_fp<16/32>bmodel_mlir.sh'脚本**

## 3. 准备模型与数据
建议使用TPU-MLIR编译BModel，Pytorch模型在编译前要导出成onnx模型。具体可参考[SAM模型导出](./docs/SAM_Export_Guide.md)。

**注意**:本例程要求TPU-MLIR版本 > v1.6.22。您可以从sftp上获取TPU-MLIR压缩包:
```bash
pip3 install dfss --upgrade

python3 -m dfss --url=open@sophgo.com:sophon-demo/SAM/tpu-mlir_v1.6.22-g62da924d-20231213.tar.gz
```


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
├── BM1684X
│   ├── decode_bmodel
│   |   ├── SAM-ViT-B_auto_multi_decoder_fp32_1b.bmodel           # decoder部分fp32 bmodel，全分割
│   │   ├── SAM-ViT-B_decoder_multi_mask_fp16_1b.bmodel     # decoder部分fp16 bmodel，输出置信度前三的mask  
│   │   ├── SAM-ViT-B_decoder_multi_mask_fp32_1b.bmodel     # decoder部分fp32 bmodel，输出置信度前三的mask  
│   │   ├── SAM-ViT-B_decoder_single_mask_fp16_1b.bmodel    # decoder部分fp16 bmodel，输出置信度第一的mask  
│   │   └── SAM-ViT-B_decoder_single_mask_fp32_1b.bmodel    # decoder部分fp32 bmodel，输出置信度第一的mask  
│   └── embedding_bmodel
│       └── SAM-ViT-B_embedding_fp16_1b.bmodel              # embedding部分fp16 bmodel
├── onnx
│   ├── decode_model_multi_mask.onnx                        # 由原模型导出的，decoder部分onnx模型，输出置信度前三的mask 
│   ├── decode_model_single_mask.onnx                       # 由原模型导出的，decoder部分onnx模型，输出置信度第一的mask 
│   ├── embedding_model.onnx                                # 由原模型导出的，embedding部分onnx模型
│   └── vit-b-auto-multi_mask.onnx                             # 由原模型导出的auto_mask_decoder部分onnx模型
└── torch
    └── sam_vit_b_01ec64.pth                                # 原torch模型

```
下载的数据包括：
```
./datasets
├── truck.jpg                                      # 测试图片1
├── groceries.jpg                                  # 测试图片2
└── dog.jpg                                        # 测试图片3         
```

## 4. 模型编译
导出的模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。建议使用TPU-MLIR编译BModel。

模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的SDK中获取)。

- 生成FP32 BModel

​本例程在`scripts`目录下提供了使用TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X**），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684x
```

​执行上述命令会在`models/BM1684X/decode_bmodel`下生成`SAM-ViT-B_decoder_multi_mask_fp32_1b.bmodel`、`SAM-ViT-B_decoder_single_mask_fp32_1b.bmodel`文件，即转换好的图像推理（mask_decoder）FP32 BModel。
**注意，目前图像压缩（embedding）不支持编译为fp32 bmodel，您可以使用fp16 bmodel进行图像压缩部分推理。**

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X**），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x
```

​执行上述命令会在`models/BM1684X/embedding_bmodel`下生成`SAM-ViT-B_embedding_fp16_1b.bmodel` 以及`models/BM1684X/decode_bmodel`下生成`SAM-ViT-B_decoder_multi_mask_fp16_1b.bmodel`、`SAM-ViT-B_decoder_single_mask_fp16_1b.bmodel`文件，即转换好的图像压缩（embedding）和图像推理（mask_decoder）FP16 BModel。

- 生成auto mask FP32 BModel

本例程在`scripts`目录下提供了TPU-MLIR编译专门用于自动掩码生成的FP32 BModel的脚本，请注意修改`gen_auto_fp32bmodel_mlir.sh`中的onnx模>型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X**），如：

```bash
./scripts/gen_auto_fp32bmodel_mlir.sh bm1684x
```

执行上述命令会在`models/BM1684X/decode_bmodel`下生成`SAM-ViT-B_auto_decoder_fp32_1b.bmodel`文件，即转换好的自动图像推理（auto_mask_decoder）FP32 BModel。


## 5. 例程测试
### 5.1 [Python例程](./python/README.md)
### 5.2 [web_ui例程](./web_ui/README.md)

## 6. 性能测试
### 6.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684X/embedding_bmodel/SAM-ViT-B_embedding_fp16_1b.bmodel
bmrt_test --bmodel models/BM1684X/decode_bmodel/SAM-ViT-B_decoder_fp32_1b.bmodel
bmrt_test --bmodel models/BM1684X/decode_bmodel/SAM-ViT-B_auto_decoder_fp32_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。

测试各个模型的理论推理时间，结果如下：

| 测试embedding/decode模型                     | calculate time(s) |
| -------------------------------------------- | ----------------- |
| SAM-ViT-B_embedding_fp16_1b.bmodel           | 0.303             |
| SAM-ViT-B_decoder_multi_mask_fp16_1b.bmodel  | 0.009             |
| SAM-ViT-B_decoder_multi_mask_fp32_1b.bmodel  | 0.027             |
| SAM-ViT-B_decoder_single_mask_fp16_1b.bmodel | 0.005             |
| SAM-ViT-B_decoder_single_mask_fp32_1b.bmodel | 0.026             |
| SAM-ViT-B_auto_decoder_fp32_1b.bmodel        | 1.503             |

> **测试说明**：  
> 1. 性能测试结果具有一定的波动性；
> 2. `calculate time`已折算为平均每张图片的推理时间；
> 3. SoC和PCIe的测试结果基本一致。

### 6.2 程序运行性能
参考[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。目前SAM_VIT仅支持1 batch的fp32和int8模型。

测试`datasets/truck.jpg`单张图片性能测试结果如下（时间单位为ms），测试结果有一定波动性：
| 测试平台    | 测试程序      | 测试模型                                                                        | decode_time | embedding_time | decode_mask_time | postprocess_time |
| ----------- | ------------- | ------------------------------------------------------------------------------- | ----------- | -------------- | ---------------- | ---------------- |
| BM1684X SoC | sam_opencv.py | SAM-ViT-B_embedding_fp16_1b.bmodel,SAM-ViT-B_decoder_multi_mask_fp16_1b.bmodel  | 11.0        | 416.0          | 15.5             | 16.2             |
| BM1684X SoC | sam_opencv.py | SAM-ViT-B_embedding_fp16_1b.bmodel,SAM-ViT-B_decoder_multi_mask_fp32_1b.bmodel  | 11.0        | 411.0          | 34.0             | 16.5             |
| BM1684X SoC | sam_opencv.py | SAM-ViT-B_embedding_fp16_1b.bmodel,SAM-ViT-B_decoder_single_mask_fp16_1b.bmodel | 11.0        | 416.0          | 15.5             | 16.2             |
| BM1684X SoC | sam_opencv.py | SAM-ViT-B_embedding_fp16_1b.bmodel,SAM-ViT-B_decoder_single_mask_fp32_1b.bmodel | 11.0        | 411.0          | 34.0             | 16.5             |
| BM1684X SoC | sam_opencv.py | SAM-ViT-B_embedding_fp16_1b.bmodel,SAM-ViT-B_auto_decoder_fp32_1b.bmodel        | 37.39       | 512.61         | 28942.54         | 9403.08          |

> **测试说明**：  
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. BM1684/1684X SoC的主控处理器均为8核 ARM A53 42320 DMIPS @2.3GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异，不同的阈值对后处理时间影响较大。

## 7. FAQ
问题请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。
