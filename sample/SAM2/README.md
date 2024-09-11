# SAM2

## 目录

- [SAM2](#sam2)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
    - [2.1 目录结构说明](#21-目录结构说明)
    - [2.2 SDK特性](#22-SDK特性)
  - [3. 准备模型与数据](#3-准备模型与数据)
    - [3.1 数据准备](#31-数据准备)
    - [3.2 模型编译](#32-模型编译)
  - [4. 例程测试](#4-例程测试)
  - [5. 精度测试](#5-精度测试)
    - [5.1 测试方法](#51-测试方法)
    - [5.2 测试结果](#52-测试结果)
  - [6. 性能测试](#6-性能测试)
    - [6.1 bmrt\_test](#61-bmrt_test)
    - [6.2 程序运行性能](#62-程序运行性能)
  
## 1. 简介

​SAM2是Meta基于SAM提出的一种实时图像和视频分割模型。SAM2适用于图像和视频。而SAM的先前版本是专门为图像使用而构建的。本例程对[​SAM2官方开源仓库](https://github.com/facebookresearch/segment-anything-2)的模型和算法进行移植，使之能在SOPHON BM1688上进行推理测试。

## 2. 特性
### 2.1 目录结构说明
```
./SAM2
├── docs
│   └── export_bmodel.md                              # 本例程中bmodel编译中的算子问题说明文档
├── pics
├── python                                             # 存放Python例程及其README
│   ├── sam2_opencv.py                                 # 基于sail和OpenCV实现的SAM2图像分割推理Python例程
│   ├── datasets.py                                    # 支持进行分割测试的数据集类
│   ├── utils.py                                       # SAM2推理流程中所用的工具类函数
│   └── README.md                                      # SAM2 Python例程的说明文件
├── README.md                                          # 本例程的中文指南
├── scripts
│   ├── auto_test.sh                                   # 自动化测试脚本
│   ├── download.sh                                    # 模型和数据集下载脚本
│   ├── gen_fp16bmodel_mlir.sh                         # F16 bmodel编译脚本
│   └── gen_fp32bmodel_mlir.sh                         # F32 bmodel编译脚本
└── tools
    └──eval.py                                         # 精度测试例程，目前只支持coco数据集
    └──compare_statis.py                               # 性能对比例程
```

### 2.2 SDK特性
* 支持BM1688(SoC)
* 图像编码器（Image Encoder）部分支持FP16、FP32的模型编译和推理，支持1core和2core
* 图像解码器（Image Decoder）部分支持FP16、FP32的模型编译和推理，支持1core和2core
* 支持基于OpenCV的Python推理
* 支持单点和box输入的模型推理，并输出满足阈值的mask
* 支持图片测试和COCO数据集测试


**注意：
本repo将图像编码器（Image Encoder）和图像解码器（Image Decoder）分为两个bmodel运行；

## 3. 数据准备与模型编译

### 3.1 数据准备

​本例程在`scripts`目录下提供了相关模型和数据的下载脚本`download.sh`，您也可以自己准备模型和数据集，并参考3.2进行模型转换。

```bash
# 安装unzip，若已安装请跳过，非ubuntu系统视情况使用yum或其他方式安装
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```

下载的模型包括：
```
./SAM2
├── BM1688
│   ├── image_decoder
│   |   ├── sam2_decoder_f16_1b_1core.bmodel           # decoder部分fp16 1core bmodel
│   |   ├── sam2_decoder_f16_1b_2core.bmodel           # decoder部分fp16 2core bmodel
│   |   ├── sam2_decoder_f32_1b_1core.bmodel           # decoder部分fp32 1core bmodel
│   │   └── sam2_decoder_f32_1b_2core.bmodel           # decoder部分fp32 2core bmodel
│   └── image_encoder
│       |── sam2_encoder_f16_1b_1core.bmodel           # encoder部分fp16 1core bmodel
│       ├── sam2_encoder_f16_1b_2core.bmodel           # encoder部分fp16 2core bmodel
│       ├── sam2_encoder_f32_1b_1core.bmodel           # encoder部分fp32 1core bmodel
|       └── sam2_encoder_f32_1b_2core.bmodel           # encoder部分fp32 2core bmodel
├── onnx
│   ├── sam2_hiera_tiny_decoder.onnx                   # 由原模型导出的，decoder部分onnx模型，输出置信度前三的mask 
│   └── sam2_hiera_tiny_encoder.onnx                   # 由原模型导出的, encoder部分onnx模型
└── torch
    └── sam2_hiera_tiny.pt                             # 原torch模型

```
下载的数据包括：
```
./datasets
├── val2017                                        # 测试集图像
├── instances_val2017.json                         # 测试集标注信息
└── truck.jpg                                      # 测试图片        
```

### 3.2 模型编译

导出的模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。建议使用TPU-MLIR编译BModel。

模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的SDK中获取)。

直接使用官方的torch模型文件进行导出和编译，会出现算子不兼容的问题，修改方法可参考[bmodel导出](docs/export_bmodel.md)

- 生成FP32 BModel

​本例程在`scripts`目录下提供了使用TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**仅支持BM1688**），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh
```

​执行上述命令会在`models/BM1688/image_encoder`下生成`sam2_encoder_f32_1b_1core.bmodel`和`sam2_encoder_f32_1b_2core.bmodel`,在`models/BM1688/image_decoder`下生成`sam2_decoder_f32_1b_1core.bmodel`及`sam2_decoder_f32_1b_2core.bmodel`文件，即转换好的图像编码和解码的单双核FP32 BModel。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了使用TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1688**），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh 
```

​执行上述命令会在`models/BM1688/image_encoder`下生成`sam2_encoder_f16_1b_1core.bmodel`和`sam2_encoder_f16_1b_2core.bmodel`,在`models/BM1688/image_decoder`下生成`sam2_decoder_f16_1b_1core.bmodel`及`sam2_decoder_f16_1b_2core.bmodel`文件，即转换好的图像编码和解码的单双核FP16 BModel。

## 4. 例程测试
### [Python例程](./python/README.md)

## 5. 精度测试
### 5.1 测试方法
参考Python例程，选择推理要测试的数据集（目前仅支持COCO数据集），生成预测的json文件，json文件会自动保存在results目录下
然后，使用tools目录下的eval.py脚本，将测试生成的json文件与测试集标签json文件进行对比，计算出目标检测的评价指标，命令如下：

```bash
python3 python/sam2_opencv.py --mode dataset --img_path datasets/val2017  --detect_num 200 --encoder_bmodel models/BM1688/image_encoder/sam2_encoder_f16_1b_2core.bmodel --decoder_bmodel models/BM1688/image_decoder/sam2_decoder_f16_1b_2core.bmodel
python3 tools/eval.py --gt_path datasets/instances_val2017.json --res_path results/sam2_encoder_f16_1b_2core_COCODataset_opencv_python_result.json
```
### 5.2 测试结果
本实例的测试方式为将COCO数据集内目标的bbox中心点作为SAM2输入的points prompt。在COCO数据集中测试选择的图像越多，mIoU的指标越高，下图中测试图像为200张。

|   测试平台    |       测试程序      |          encoder_bmodel         |           decoder_bmodel        |   mIoU   |
|  ----------- |------------------- |--------------------------------- |-------------------------------- |-------- |
| SE9-16       | sam2_opencv.py     | sam2_encoder_f32_1b_1core.bmodel | sam2_decoder_f32_1b_1core.bmodel|    0.472|
| SE9-16       | sam2_opencv.py     | sam2_encoder_f32_1b_2core.bmodel | sam2_decoder_f32_1b_2core.bmodel|    0.472|
| SE9-16       | sam2_opencv.py     | sam2_encoder_f16_1b_1core.bmodel | sam2_decoder_f16_1b_1core.bmodel|    0.468|
| SE9-16       | sam2_opencv.py     | sam2_encoder_f16_1b_2core.bmodel | sam2_decoder_f16_1b_2core.bmodel|    0.468|

## 6. 性能测试
### 6.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径
bmrt_test --bmodel models/BM1688/image_decoder/sam2_decoder_f16_1b_1core.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。

测试各个模型的理论推理时间，结果如下：

| 测试模型                                                | calculate time(ms) |
| ------------------------------------------------------------|  -------------- |
| BM1688/image_encoder/sam2_encoder_f16_1b_1core.bmodel       |         373.71  |
| BM1688/image_encoder/sam2_encoder_f16_1b_2core.bmodel       |         225.64  |
| BM1688/image_encoder/sam2_encoder_f32_1b_1core.bmodel       |        2248.32  |
| BM1688/image_encoder/sam2_encoder_f32_1b_2core.bmodel       |        1319.28  |
| BM1688/image_decoder/sam2_decoder_f16_1b_1core.bmodel       |          11.72  |
| BM1688/image_decoder/sam2_decoder_f16_1b_2core.bmodel       |           8.73  |
| BM1688/image_decoder/sam2_decoder_f32_1b_1core.bmodel       |          47.90  |
| BM1688/image_decoder/sam2_decoder_f32_1b_2core.bmodel       |          31.55  |

> **测试说明**：  
> 1. 性能测试结果具有一定的波动性；
> 2. `calculate time`已折算为平均每张图片的推理时间；

### 6.2 程序运行性能
参考[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。目前SAM2仅支持1 batch的fp32和fp16模型。前处理部分包含对图像、points和bbox等的处理操作，后处理部分仅对包含mask的简单处理。

测试`datasets/truck.jpg`单张图片性能测试结果如下（时间单位为ms），测试结果有一定波动性：

|  测试平台   |      测试程序       |          encoder_bmodel           |          decoder_bmodel           | preprocess_time |  encoder_time   |  decoder_time  | postprocess_time |
|----------|----------|----------|----------|----------|----------|----------|----------|
|   SE9-16    |  sam2_opencv.py   | sam2_encoder_f32_1b_1core.bmodel  | sam2_decoder_f32_1b_1core.bmodel  |      95.91      |     2394.54     |      74.43      |      1.07       |
|   SE9-16    |  sam2_opencv.py   | sam2_encoder_f32_1b_2core.bmodel  | sam2_decoder_f32_1b_2core.bmodel  |      99.32      |     1472.30     |      58.43      |      1.14       |
|   SE9-16    |  sam2_opencv.py   | sam2_encoder_f16_1b_1core.bmodel  | sam2_decoder_f16_1b_1core.bmodel  |      96.10      |     457.77      |      37.05      |      2.77       |
|   SE9-16    |  sam2_opencv.py   | sam2_encoder_f16_1b_2core.bmodel  | sam2_decoder_f16_1b_2core.bmodel  |     100.79      |     311.52      |      34.60      |      1.11       |

> **测试说明**：  
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异，不同的阈值对后处理时间影响较大。
