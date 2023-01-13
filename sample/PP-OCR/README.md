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

PP-OCRv2，是百度飞桨团队开源的超轻量OCR系列模型，包含文本检测、文本分类、文本识别模型，是PaddleOCR工具库的重要组成之一。支持中英文数字组合识别、竖排文本识别、长文本识别，其性能及精度较之前的PP-OCR版本均有明显提升。本例程对[PaddleOCR-release-2.4](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.4)的模型和算法进行移植，使之能在SOPHON BM1684和BM1684X上进行推理测试。

## 2. 数据集

PP-OCRv2开源仓库提供部分测试图片，本例程使用开源仓库提供的测试图片进行测试。可通过例程提供的脚本进行下载。

PP-OCRv2开源仓库提供提供的更多公开数据集：
- [通用中英文OCR数据集](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.4/doc/doc_ch/datasets.md)
- [手写中文OCR数据集](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.4/doc/doc_ch/handwritten_datasets.md)
- [垂类多语言OCR数据集](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.4/doc/doc_ch/vertical_and_multilingual_datasets.md)

## 3. 准备模型与数据
您需要准备用于测试的数据集，如果量化模型，还要准备用于量化的数据集。

本例程在`scripts`目录下提供了相关模型和数据集的下载脚本`download.sh`，您也可以自己准备模型和数据集，并参考[4. 模型转换](#4-模型转换)进行模型转换。
```bash
sudo chmod -R +x scripts/
./scripts/download.sh
```
执行后，模型保存至`data/models/`，数据集下载并解压至`data/images/`

下载的模型包括：
```
paddle/ch_PP-OCRv2_det_infer: 原始文本检测inference模型
paddle/ch_ppocr_mobile_v2.0_cls_infer: 原始方向分类器inference模型
paddle/ch_PP-OCRv2_rec_infer: 原始文字识别inference模型

## BM1684上测试使用的模型文件：

文本检测FP32Bmodel模型文件：
用于组合ch_PP-OCRv2_det_fp32_b1b4.bmodel的模型：
BM1684/ch_PP-OCRv2_det_1b.bmodel
BM1684/ch_PP-OCRv2_det_4b.bmodel
通过 ch_PP-OCRv2_det_1b.bmodel 和 ch_PP-OCRv2_det_4b.bmodel 组合得到：
BM1684/ch_PP-OCRv2_det_fp32_b1b4.bmodel: 用于BM1684的FP32 BModel，支持使用batch_size=1或4的输入

方向分类器FP32Bmodel模型文件：
用于组合ch_ppocr_mobile_v2.0_cls_fp32_b1b4.bmodel的模型：
BM1684/ch_ppocr_mobile_v2.0_cls_1b.bmodel
BM1684/ch_ppocr_mobile_v2.0_cls_4b.bmodel
通过 ch_ppocr_mobile_v2.0_cls_1b.bmodel 和 ch_ppocr_mobile_v2.0_cls_4b.bmodel 组合得到：
BM1684/ch_ppocr_mobile_v2.0_cls_fp32_b1b4.bmodel: 用于BM1684的FP32 BModel，支持使用batch_size=1或4的输入

文字识别FP32Bmodel模型文件：
BM1684/ch_PP-OCRv2_rec_320_1b.bmodel
BM1684/ch_PP-OCRv2_rec_320_4b.bmodel
BM1684/ch_PP-OCRv2_rec_640_1b.bmodel
BM1684/ch_PP-OCRv2_rec_640_4b.bmodel
BM1684/ch_PP-OCRv2_rec_1280_1b.bmodel
通过以上模型组合得到，处理不同图片尺寸的文字识别模型：
BM1684/ch_PP-OCRv2_rec_fp32_b1b4.bmodel: 用于BM1684的FP32 BModel，支持使用batch_size=1或4的输入

## BM1684X上测试使用的模型文件：

文本检测FP32Bmodel模型文件：
BM1684X/ch_PP-OCRv2_det_1b.bmodel: 用于BM1684X的FP32 BModel，支持使用batch_size=1的输入

方向分类器FP32Bmodel模型文件：
BM1684X/ch_ppocr_mobile_v2.0_cls_1b.bmodel: 用于BM1684X的FP32 BModel，支持使用batch_size=1的输入

文字识别FP32Bmodel模型文件：
BM1684X/ch_PP-OCRv2_rec_320_1b.bmodel
BM1684X/ch_PP-OCRv2_rec_640_1b.bmodel
BM1684X/ch_PP-OCRv2_rec_1280_1b.bmodel
通过以上模型组合得到，处理不同图片尺寸的文字识别模型：
BM1684X/ch_PP-OCRv2_rec_fp32_b1.bmodel: 用于BM1684X的FP32 BModel，支持使用batch_size=1的输入
```

模型信息：
| 原始模型 | ch_PP-OCRv2_det_infer | 
| ------- | ------------------------------  |
| 概述     | 检测模型 | 
| 骨干网络 | ResNet | 
| 训练集   | ICDAR2015 TextLocalization数据集 | 
| 输入数据 | [batch_size, 3, 960, 960], FP32, NCHW |
| 输出数据 | [batch_size, 1, 960, 960], FP32, NCHW |
| 前处理   | resize, 减均值，除方差，填充 |
| 后处理   | resize, Differentiable Binarization (DB), filter, sort, clip|

模型信息：
| 原始模型 | ch_ppocr_mobile_v2.0_cls_infer | 
| ------- | ------------------------------  |
| 概述     | 方向分类器模型 | 
| 骨干网络 | MobileNetV3 | 
| 输入数据 | [batch_size, 3, 48, 192], FP32, NCHW |
| 输出数据 | [batch_size, 2], FP32 |
| 前处理   | resize, 减均值，除方差，填充 |
| 后处理   | argmax, 获得置信度最大的文字方向分类 |

模型信息：
| 原始模型 | ch_PP-OCRv2_rec_infer | 
| ------- | ------------------------------  |
| 概述     | 识别模型 | 
| 骨干网络 | MobileNetV1Enhance  | 
| 训练集   | ICDAR2015 | 
| 输入数据 | [batch_size, 3, 32, 320/640/1280], FP32, NCHW |
| 输出数据 | [batch_size, 80/160/320, 6625], FP32 |
| 前处理   | resize, 减均值，除方差，填充 |
| 后处理   | argmax, 获得置信度最大的字符索引, 获得识别的文本及置信度 |



下载的数据包括：
```
ppocr_img: 用于测试的数据集
```

## 4. 模型转换
模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。

BMPADDLE是针对paddlepaddle的模型编译器，可以将inference模型文件编译成BMRuntime所需的文件。需要注意的是，在使用BMPADDLE编译BModel模型前需要将的训练好的checkpoints模型转换为inference模型。

inference 模型（paddle.jit.save保存的模型） 一般是模型训练，把模型结构和模型参数保存在文件中的固化模型，多用于预测部署场景。 训练过程中保存的模型是checkpoints模型，保存的只有模型的参数，多用于恢复训练等。 与checkpoints模型相比，inference 模型会额外保存模型的结构信息，在预测部署、加速推理上性能优越，灵活方便，适合于实际系统集成。可以通过PP-OCRv2开源仓库提供的文档[《训练模型转inference模型》](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.4/doc/doc_ch/inference.md)进行转换。

模型编译前需要安装TPU-NNTC(>=3.1.0)，具体可参考[tpu-nntc环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-nntc环境搭建)。

### 4.1 生成fp32 bmodel
模型编译为FP32 BModel，具体方法可参考TPU-NNTC开发参考手册。

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
