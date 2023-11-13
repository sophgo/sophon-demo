[简体中文](./README.md) | [English](./README_EN.md)

# SegFormer

* [1. 简介](#1-简介)
* [2. 特性](#2-特性)
* [3. 准备模型与数据](#3-准备模型与数据)
* [4. 模型编译](#4-模型编译)
* [5. 例程测试](#5-例程测试)

## 1. 简介
SegFormer是一种用于语义分割的简单、高效和强大的方法。SegFormer使用了Transformer技术，Transformer是一种用于序列建模的深度学习模型，它在自然语言处理中广泛应用。本例程对[​SegFormer官方开源仓库](https://github.com/NVlabs/SegFormer)版本的模型和算法进行移植，使之能在SOPHON BM1684和BM1684X上进行推理测试。

## 2. 特性
* 支持BM1684X(x86 PCIe、SoC)和BM1684(x86 PCIe、SoC、arm PCIe)
* 支持FP32、FP16(BM1684X)、INT8模型编译和推理
* 支持基于BMCV预处理的C++推理
* 支持基于OpenCV和BMCV预处理的Python推理
* 支持图片和视频测试

## 3. 准备模型与数据
如果您使用BM1684芯片，建议使用TPU-NNTC编译BModel，目前官方的Segfomer只有pth预训练模型，pth模型在编译前要导出成onnx模型；如果您使用BM1684X芯片，建议使用TPU-MLIR编译BModel，pth模型在编译前要导出成onnx模型。具体可参考[Segformer模型导出](./docs/Segformer_Export_Guide.md)。

同时，您需要准备用于测试的数据集，如果量化模型，还要准备用于量化的数据集。

本例程使用[Cityscapes](https://www.cityscapes-dataset.com/downloads/)进行测试，更多的公开数据集，请参考官方推荐[Prepare datasets](https://github.com/NVlabs/SegFormer/blob/master/docs/dataset_prepare.md)

​本例程在`scripts`目录下提供了相关模型和数据的下载脚本`download.sh`，您也可以自己准备模型和数据集，并参考[4. 模型编译](#4-模型编译)进行模型转换。

```bash
# 安装unzip，若已安装请跳过，非ubuntu系统视情况使用yum或其他方式安装
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```
下载的模型包括：
./models
├── BM1684
│   ├── segformer.b0.512x1024.city.160k_fp32_1b.bmodel # 使用TPU-MLIR编译，用于BM1684的FP32 BModel，batch_size=1
├── BM1684X
│   ├── segformer.b0.512x1024.city.160k_fp32_1b.bmodel # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=1
│   ├── segformer.b0.512x1024.city.160k_fp16_1b.bmodel # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=1
└── onnx
    └── segformer.b0.512x1024.city.160k.onnx # pt导出的onnx动态模型

下载的数据包括：
./datasets
├── cali                                #量化图片
│   ├── xxx.png                                                                                 
├── cityscapes                          #测试图片集
│   ├── gtFine                          #评价图片 
│   ├── leftImg8bit                     #测试图片                
│   └── val.txt                         #评价图片列表
├── cityscapes_small                    #测试图片集—小
│   ├── gtFine                          #评价图片
│   └── leftImg8bit                     #测试图片
└── cityscapes_video.avi           #测试视频


## 4. 模型编译
导出的模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。建议使用TPU-MLIR编译BModel。

模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#2-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index/material/31/all.html)相应版本的SDK中获取)。

- 生成FP32 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684/BM1684X**），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh BM1684
#or
./scripts/gen_fp32bmodel_mlir.sh BM1684X
```

​执行上述命令会在`models/BM1684`或`models/BM1684X/`下生成`segformer.b0.512x1024.city.160k_fp32_1b.bmodel`等文件，即转换好的FP32 BModel。


- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X**），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh BM1684X
```

​执行上述命令会在`models/BM1684X/`下生成`segformer.b0.512x1024.city.160k_fp16_1b.bmodel`等文件，即转换好的FP16 BModel。



## 5. 推理测试
* [C++例程](cpp/README.md)
* [Python例程](python/README.md)


## 6. 精度测试
### 6.1 测试方法
首先，参考[C++例程](cpp/README.md#32-测试图片)或[Python例程](python/README.md#22-测试图片)推理要测试的数据集，生成预测的json文件，注意修改数据集(datasets/cityscapes)。
然后，使用`tools`目录下的`segformer_eval.py`脚本，将测试生成的json文件与测试集标签json文件进行对比，计算出目标检测的评价指标，命令如下：

```bash
# 请根据实际情况修改程序路径和json文件路径
python3 tools/segformer_eval.py --result_json python/results/segformer.b0.512x1024.city.160k_fp32_1b.bmodel_cityscapes_opencv_python_result.json
python3 tools/segformer_eval.py --result_json cpp/segformer_bmcv/results/segformer.b0.512x1024.city.160k_fp32_1b.bmodel_cityscapes_sail_cpp_result.json
```

### 6.2 测试结果
采用1684 fp32模型在cityscapes数据集上，其精度如下精度测试结果如下：
|   测试平台    |      测试程序        |                     测试模型                   |  mIoU | mAcc  | aAcc  |
| ------------ | ------------------- | ---------------------------------------------- | ----- | ----- |-------|
| BM1684 PCIe  | segformer_opencv.py | segformer.b0.512x1024.city.160k_fp32_1b.bmodel | 68.35 | 76.96 | 94.75 |
| BM1684 PCIe  | segformer_bmcv.py   | segformer.b0.512x1024.city.160k_fp32_1b.bmodel | 68.05 | 76.60 | 94.68 |
| BM1684 PCIe  | segformer_bmcv.cpp  | segformer.b0.512x1024.city.160k_fp32_1b.bmodel | 68.33 | 76.94 | 94.75 |
| BM1684 PCIe  | segformer_sail.cpp  | segformer.b0.512x1024.city.160k_fp32_1b.bmodel | 68.24 | 76.82 | 94.73 |
| BM1684X PCIe | segformer_opencv.py | segformer.b0.512x1024.city.160k_fp32_1b.bmodel | 68.35 | 76.96 | 94.75 |
| BM1684X PCIe | segformer_bmcv.py   | segformer.b0.512x1024.city.160k_fp32_1b.bmodel | 68.04 | 76.58 | 94.68 |
| BM1684X PCIe | segformer_bmcv.pcie | segformer.b0.512x1024.city.160k_fp32_1b.bmodel | 68.35 | 76.96 | 94.75 |
| BM1684X PCIe | segformer_sail.pcie | segformer.b0.512x1024.city.160k_fp32_1b.bmodel | 68.35 | 76.96 | 94.75 |
| BM1684X PCIe | segformer_opencv.py | segformer.b0.512x1024.city.160k_fp16_1b.bmodel | 68.34 | 76.95 | 94.75 |
| BM1684X PCIe | segformer_bmcv.py   | segformer.b0.512x1024.city.160k_fp16_1b.bmodel | 68.02 | 76.53 | 94.68 |
| BM1684X PCIe | segformer_bmcv.pcie | segformer.b0.512x1024.city.160k_fp16_1b.bmodel | 68.35 | 76.96 | 94.75 |
| BM1684X PCIe | segformer_sail.pcie | segformer.b0.512x1024.city.160k_fp16_1b.bmodel | 68.34 | 76.96 | 94.75 |

> **测试说明**：  
> 1. batch_size=4和batch_size=1的模型精度一致；
> 2. SoC和PCIe的模型精度一致；

## 7. 性能测试
### 7.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684/segformer.b0.512x1024.city.160k_fp32_1b.bmodel
```
在cityscapes测试各个模型的理论推理时间，结果如下：
|                  测试模型            |stage| calculate time(ms) |
| ----------------------------------  |  ---| ----------------- |
| BM1684/segformer.b0.512x1024.city.160k_fp32_1b.bmodel   |  1  | 370.067         |
| BM1684X/segformer.b0.512x1024.city.160k_fp32_1b.bmodel  |  1  | 288.866         |
| BM1684X/segformer.b0.512x1024.city.160k_fp16_1b.bmodel  |  1  | 54.229          |

> **测试说明**：  
> 1. 性能测试结果具有一定的波动性；
> 2. `calculate time`已折算为平均每张图片的推理时间；
> 3. SoC和PCIe的测试结果基本一致。


### 7.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++例程打印的预处理时间、推理时间、后处理时间为整个batch处理的时间，需除以相应的batch size才是每张图片的处理时间。

在不同的测试平台上，使用不同的例程、模型测试`datasets/cityscapes`,性能测试结果如下：

|    测试平台   |       测试程序      |                       测试模型                  |decode_time|preprocess_time|inference_time |postprocess_time| 
| -----------  | ------------------- | ---------------------------------------------- | --------- | ------------- | ------------- | -------------- |
| BM1684 SoC   | segformer_opencv.py | segformer.b0.512x1024.city.160k_fp32_1b.bmodel | 109.20    | 25.02         | 391.65        | 186.31         |
| BM1684 SoC   | segformer_bmcv.py   | segformer.b0.512x1024.city.160k_fp32_1b.bmodel | 353.90    | 5.62          | 369.59        | 141.78         |
| BM1684 SoC   | segformer_bmcv.soc  | segformer.b0.512x1024.city.160k_fp32_1b.bmodel | 114.52    | 1.38          | 364.12        | 261.13         | 
| BM1684 SoC   | segformer_sail.soc  | segformer.b0.512x1024.city.160k_fp32_1b.bmodel | 370.24    | 6.89          | 365.85        | 256.61         |
| BM1684X SoC  | segformer_opencv.py | segformer.b0.512x1024.city.160k_fp32_1b.bmodel | 109.69    | 30.40         | 335.98        | 178.17         |
| BM1684X SoC  | segformer_bmcv.py   | segformer.b0.512x1024.city.160k_fp32_1b.bmodel | 355.00    | 5.15          | 317.73        | 126.43         |
| BM1684X SoC  | segformer_bmcv.soc  | segformer.b0.512x1024.city.160k_fp32_1b.bmodel | 114.30    | 1.39          | 313.07        | 265.03         |
| BM1684X SoC  | segformer_sail.soc  | segformer.b0.512x1024.city.160k_fp32_1b.bmodel | 0.027     | 7.91          | 313.40        | 260.65         |
| BM1684X SoC  | segformer_opencv.py | segformer.b0.512x1024.city.160k_fp32_1b.bmodel | 109.69    | 30.40         | 335.98        | 178.17         |
| BM1684X SoC  | segformer_bmcv.py   | segformer.b0.512x1024.city.160k_fp32_1b.bmodel | 355.00    | 5.15          | 317.73        | 126.43         |
| BM1684X SoC  | segformer_bmcv.soc  | segformer.b0.512x1024.city.160k_fp32_1b.bmodel | 114.56    | 1.39          | 69.45         | 260.90         |
| BM1684X SoC  | segformer_sail.soc  | segformer.b0.512x1024.city.160k_fp32_1b.bmodel | 0.027     | 7.91          | 69.77         | 260.36         |


> **测试说明**：  
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. BM1684/1684X SoC的主控CPU均为8核 ARM A53 42320 DMIPS @2.3GHz，PCIe上的性能由于CPU的不同可能存在较大差异；
> 4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异。 

## 8. FAQ
其他问题请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。
