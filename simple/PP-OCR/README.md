# PP-OCR

## 目录
* [PP-OCR](#PP-OCR)
  * [目录](#目录)
  * [1. 简介](#1-简介)
  * [2. 数据集](#2-数据集)
  * [3. 准备模型与数据](#3-准备模型与数据)
  * [4. 模型编译](#4-模型编译)
    * [4.1 生成FP32 BModel](#41-生成fp32-bmodel)
    * [4.2 生成INT8 BModel](#42-生成int8-bmodel)
  * [5. 例程测试](#5-例程测试)
    
## 1. 简介

PP-OCR，是百度飞桨团队开源的超轻量OCR系列模型，包含文本检测、文本分类、文本识别模型，是PaddleOCR工具库的重要组成之一。支持中英文数字组合识别、竖排文本识别、长文本识别，其性能及精度较PP-OCR均有明显提升。本例程对[PaddleOCR-release-2.4](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.4)的模型和算法进行移植，使之能在SOPHON BM1684和BM1684X上进行推理测试。

仓库链接：https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.4

## 2. 数据集

暂未找到PP-OCR系列模型对应的数据集。

## 3. 准备环境与数据
您需要准备用于测试的数据集，如果量化模型，还要准备用于量化的数据集。

本例程在`scripts`目录下提供了相关模型和数据集的下载脚本`download.sh`，您也可以自己准备模型和数据集，并参考[4. 模型转换](#4-模型转换)进行模型转换。
```bash
sudo chmod +x scripts/
./scripts/download.sh
```
执行后，模型保存至`data/models`，数据集下载并解压至`data/images/`

下载的模型包括：

ch_PP-OCRv2_det_infer: 原始检测模型

ch_ppocr_mobile_v2.0_cls_infer: 原始方向分类器

ch_PP-OCRv2_rec_infer: 原始识别模型

BM1684/ch_PP-OCRv2_det_1b.bmodel: 用于BM1684的FP32 BModel，batch_size=1
BM1684/ch_PP-OCRv2_det_4b.bmodel: 用于BM1684的FP32 BModel，batch_size=4
BM1684/ch_PP-OCRv2_det_fp32_b1b4.bmodel: 用于BM1684的FP32 BModel，batch_size=1,4
BM1684/ch_ppocr_mobile_v2.0_cls_1b.bmodel: 用于BM1684的INT8 BModel，batch_size=1
BM1684/ch_ppocr_mobile_v2.0_cls_4b.bmodel: 用于BM1684的INT8 BModel，batch_size=4
BM1684/ch_ppocr_mobile_v2.0_cls_fp32_b1b4.bmodel: 用于BM1684的FP32 BModel，batch_size=1,4
BM1684/ch_PP-OCRv2_rec_320_1b.bmodel: 用于BM1684的INT8 BModel，batch_size=1
BM1684/ch_PP-OCRv2_rec_320_4b.bmodel: 用于BM1684的INT8 BModel，batch_size=4
BM1684/ch_PP-OCRv2_rec_640_1b.bmodel: 用于BM1684的INT8 BModel，batch_size=1
BM1684/ch_PP-OCRv2_rec_640_4b.bmodel: 用于BM1684的INT8 BModel，batch_size=4
BM1684/ch_PP-OCRv2_rec_1280_4b.bmodel: 用于BM1684的INT8 BModel，batch_size=1
BM1684/ch_PP-OCRv2_rec_fp32_b1b4.bmodel: 用于BM1684的FP32 BModel，batch_size=1,4

BM1684X/ch_PP-OCRv2_det_1b.bmodel: 用于BM1684X的FP32 BModel，batch_size=1
BM1684X/ch_PP-OCRv2_det_4b.bmodel: 用于BM1684X的FP32 BModel，batch_size=4
BM1684X/ch_PP-OCRv2_det_fp32_b1b4.bmodel: 用于BM1684X的FP32 BModel，batch_size=1,4
BM1684X/ch_ppocr_mobile_v2.0_cls_1b.bmodel: 用于BM1684X的INT8 BModel，batch_size=1
BM1684X/ch_ppocr_mobile_v2.0_cls_4b.bmodel: 用于BM1684X的INT8 BModel，batch_size=4
BM1684X/ch_ppocr_mobile_v2.0_cls_fp32_b1b4.bmodel: 用于BM1684X的FP32 BModel，batch_size=1,4
BM1684X/ch_PP-OCRv2_rec_320_1b.bmodel: 用于BM1684X的INT8 BModel，batch_size=1
BM1684X/ch_PP-OCRv2_rec_320_4b.bmodel: 用于BM1684X的INT8 BModel，batch_size=4
BM1684X/ch_PP-OCRv2_rec_640_1b.bmodel: 用于BM1684X的INT8 BModel，batch_size=1
BM1684X/ch_PP-OCRv2_rec_640_4b.bmodel: 用于BM1684X的INT8 BModel，batch_size=4
BM1684X/ch_PP-OCRv2_rec_1280_4b.bmodel: 用于BM1684X的INT8 BModel，batch_size=1
BM1684X/ch_PP-OCRv2_rec_fp32_b1b4.bmodel: 用于BM1684X的FP32 BModel，batch_size=1,4

下载的数据包括：
ppocr_img: 用于测试的数据集

模型信息：
| 原始模型 | ch_PP-OCRv2_det_infer | 
| ------- | ------------------------------  |
| 概述     | 检测模型 | 
| 骨干网络 | MobileNetV3、ResNet_vd等 | 
| 训练集   | icdar2015 TextLocalization数据集 | 
| 输入数据 | [batch_size, 3, 960, 960], FP32，NCHW |
| 输出数据 | [batch_size, 960, 960], FP32 |
| 前处理   | resize |
| 后处理   | Differentiable Binarization |

模型信息：
| 原始模型 | ch_ppocr_mobile_v2.0_cls_infer | 
| ------- | ------------------------------  |
| 概述     | 方向分类器模型 | 
| 输入数据 | [batch_size, 3, 48, 192], FP32，NCHW |
| 输出数据 | 预测的标签和置信度 |

模型信息：
| 原始模型 | ch_PP-OCRv2_rec_infer | 
| ------- | ------------------------------  |
| 概述     | 识别模型 | 
| 骨干网络 | CRNN等  | 
| 训练集   | ICDAR2015, icdar2013, icdar2015, cocotext, IIIT5 | 
| 输入数据 | [batch_size, 3, 32, 320/640/1280], FP32，NCHW |
| 输出数据 | 预测的文本内容及置信度 |

## 4. 模型转换
模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。
模型编译前需要安装tpu-nntc，具体可参考[TPU-NNTC开发参考手册]()。

### 4.1 生成fp32 bmodel
模型编译为FP32 BModel，具体方法可参考[BMPADDLE 使用]()。

本例程在`scripts`目录下提供了编译FP32 BModel的脚本。请注意修改`gen_fp32bmodel.sh`中的模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（支持BM1684和BM1684X），如：

```bash
./scripts/gen_fp32bmodel.sh BM1684
```
执行上述命令会在`data/models/BM1684/`下生成`ch_PP-OCRv2_det_infer, ch_ppocr_mobile_v2.0_cls_infer, ch_PP-OCRv2_rec_infer`中的paddle模型对应的FP32 BModel的文件。

### 4.2 生成INT8 BModel
TODO 

不量化模型可跳过本节。

## 5. 推理测试
* [Python例程](python/README.md)
