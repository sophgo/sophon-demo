# RetinaFace

## 目录

- [RetinaFace](#RetinaFace)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 数据集](#2-数据集)
  - [3. 准备环境与数据](#3-准备环境与数据)
  - [4. 模型转换](#4-模型转换)
  - [5. 例程测试](#5-例程测试)
  - [6. 性能测试](#6-性能测试)

## 1. 简介
本例程对[Retinaface]的模型和算法进行移植，使之能在SOPHON BM1684和BM1684X上进行推理测试。


**论文:** [Retinaface论文](https://arxiv.org/pdf/1905.00641.pdf)
RetinaFace: Single-stage Dense Face Localisation in the Wild，提出了一种鲁棒的single stage人脸检测器，名为RetinaFace，它利用额外监督(extra-supervised)和自监督(self-supervised)结合的多任务学习(multi-task learning)，对不同尺寸的人脸进行像素级定位。具体来说，Retinaface在以下五个方面做出了贡献：

(1) 在WILDER FACE数据集中手工标注了5个人脸关键点（Landmark），并在这个额外的监督信号的帮助下，观察到在face检测上的显著改善。

(2) 进一步添加自监督网络解码器(mesh decoder)分支，与已有的监督分支并行预测像素级的3D形状的人脸信息。

(3) 在IJB-C测试集中，RetinaFace使state-of-the-art 方法(Arcface)在人脸识别中的结果得到提升（FAR=1e6，TAR=85.59%）。

(4) 采用轻量级的backbone 网络，RetinaFace能在单个CPU上实时运行VGA分辨率的图像。


## 2. 数据集
[Retinaface论文](https://arxiv.org/abs/1806.10447v1) 使用的数据集为WIDER FACE数据集。WIDER FACE数据集由 32,203 个图像和 393,703 个人脸边界框组成。WIDER FACE 数据集通过从61个场景类别中随机抽样分为训练 (40%)、验证 (10%) 和测试 (50%) 子集。
WIDER FACE下载地址：http://shuoyang1213.me/WIDERFACE/


## 3. 准备模型与数据
您需要准备用于测试的数据集。

本例程在`scripts`目录下提供了相关模型和数据集的下载脚本`download.sh`，您也可以自己准备模型和数据集，并参考[4. 模型转换](#4-模型转换)进行模型转换。
```bash
chmod -R +x scripts/
./scripts/download.sh
```
执行后，模型保存至`data/models`，图片数据集下载并解压至`data/images/`， 视频保存至`data/videos/`
```
下载的模型包括：
onnx/retinaface_mobilenet0.25.onnx: 原始模型
BM1684/retinaface_mobilenet0.25_fp32_1b.bmodel: 用于BM1684的FP32 BModel，batch_size=1
BM1684/retinaface_mobilenet0.25_int8_1b.bmodel: 用于BM1684的INT8 BModel，batch_size=1
BM1684/retinaface_mobilenet0.25_int8_4b.bmodel: 用于BM1684的INT8 BModel，batch_size=4
BM1684X/retinaface_mobilenet0.25_fp32_1b.bmodel: 用于BM1684X的FP32 BModel，batch_size=1
BM1684X/retinaface_mobilenet0.25_fp16_1b.bmodel: 用于BM1684X的FP16 BModel，batch_size=1
BM1684X/retinaface_mobilenet0.25_int8_1b.bmodel: 用于BM1684X的INT8 BModel，batch_size=1
BM1684X/retinaface_mobilenet0.25_int8_4b.bmodel: 用于BM1684X的INT8 BModel，batch_size=4

下载的数据包括：
WIDERVAL: 测试集
face01.jpg-face05.jpg:测试图片
station.avi: 测试视频

```
模型信息：

| 原始模型 | retinaface_mobilenet0.25.onnx | 
| ------- | ------------------------------  |
| 概述     | 人脸检测模型 | 
| 骨干网络 |  mobilenet0.25  ResNet50 | 
| 训练集   |  WiderFace | 
| 输入数据 | [batch_size, 3, 640, 640], FP32，NCHW |
| 输出数据 | [batch_size, 68, 18], FP32 |
| 前处理   | resize,减均值,除方差,HWC->CHW |
| 后处理   | filter  NMS |

请注意，该onnx所用版本为1.6.0，若环境中onnx版本过高，可能会出现编译失败的现象。

## 4. 模型转换
导出的模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。建议使用TPU-MLIR编译BModel。

模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index/material/31/all.html)相应版本的SDK中获取)。

- 生成FP32 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684/BM1684X**），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684
#or
./scripts/gen_fp32bmodel_mlir.sh bm1684x
```

​执行上述命令会在`data/models/BM1684`或`data/models/BM1684X/`下生成`retinaface_mobilenet0.25_fp32_1b.bmodel`文件，即转换好的FP32 BModel。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X**），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x
```

​执行上述命令会在`data/models/BM1684X/`下生成`retinaface_mobilenet0.25_fp16_1b.bmodel`文件，即转换好的FP16 BModel。

- 生成INT8 BModel

​本例程在`scripts`目录下提供了两种量化INT8 BModel的脚本，请注意修改`gen_int8bmodel_mlir_qtable.sh`和`gen_int8bmodel_mlir_sensitive_layer.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台（**支持BM1684/BM1684X**），如：

```shell
./scripts/gen_int8bmodel_mlir_qtable.sh bm1684
./scripts/gen_int8bmodel_mlir_sensitive_layer.sh bm1684
#或
./scripts/gen_int8bmodel_mlir_qtable.sh bm1684x
./scripts/gen_int8bmodel_mlir_sensitive_layer.sh bm1684x
```

本例程在量化INT8 BModel使用了混合精度，可以修改`gen_int8bmodel_mlir_sensitive_layer.sh`中`--max_float_layers`参数或修改`gen_int8bmodel_mlir_qtable.sh`中`head`参数进一步提高模型精度，`head`参数可以提高后面参数值来使用更多的fp层。

​上述脚本会在`data/models/BM1684`或`data/models/BM1684X/`下生成`retinaface_mobilenet0.25_int8_1b.bmodel`和`retinaface_mobilenet0.25_int8_4b.bmodel`等文件，即转换好的INT8 BModel。

## 5. 例程测试
* [C++例程](cpp/README.md)
* [Python例程](python/README.md)

## 6. 性能测试
可以使用bmrt_test测试模型的理论性能：
```bash
bmrt_test --bmodel {path_of_bmodel}
```
也可以参考[5. 例程测试](#5-例程测试)打印程序运行中的实际性能指标。  
测试中性能指标存在一定的波动属正常现象。

测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

|                      测试模型                    | calculate time(ms) |
| ----------------------------------------------- | ----------------- |
| BM1684/retinaface_mobilenet0.25_fp32_1b.bmodel  | 6.31               |
| BM1684/retinaface_mobilenet0.25_int8_1b.bmodel  | 7.40               |
| BM1684/retinaface_mobilenet0.25_int8_4b.bmodel  | 4.75               |
| BM1684X/retinaface_mobilenet0.25_fp32_1b.bmodel | 3.81               |
| BM1684X/retinaface_mobilenet0.25_fp16_1b.bmodel | 1.35               |
| BM1684X/retinaface_mobilenet0.25_int8_1b.bmodel | 1.22               |
| BM1684X/retinaface_mobilenet0.25_int8_4b.bmodel | 1.05               |

> **测试说明**：  
> 1. 性能测试结果具有一定的波动性；
> 2. `calculate time`已折算为平均每张图片的推理时间；
> 3. SoC和PCIe的测试结果基本一致。
