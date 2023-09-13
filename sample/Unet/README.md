# Unet

## 目录

- [Unet](#unet)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 数据集](#2-数据集)
  - [3. 准备模型与数据](#3-准备模型与数据)
  - [4. 模型转换](#4-模型转换)
    - [4.1 生成fp32 bmodel](#41-生成fp32-bmodel)
  - [5. 例程测试](#5-例程测试)
  - [6. 精度测试](#6-精度测试)

## 1. 简介
本例程对[Unet]的模型和算法进行移植，使之能在SOPHON BM1684上进行推理测试。


**论文:** [Unet论文](https://arxiv.org/pdf/1505.04597.pdf)
U-Net: Convolutional Networks for Biomedical Image Segmentation，是2015年于MICCAI发表的经典的语义分割网络。原论文在医学影像领域对其性能进行了测试，但它也可以应用到更加广泛的任务上。

## 2. 数据集
本例程使用Unet对Carvana汽车数据集进行分割。Carvana是一个在线二手车销售公司,由于他们拍摄的车辆照片中车辆与背景颜色相近，使用一般的工具提取汽车区域时往往会出现分割错误的问题。Carvana数据集由Kaggle提供，其中train.zip和train_masks.zip包括5088张汽车图像对应的标注。
Carvana数据集下载地址:

图像：https://www.kaggle.com/competitions/carvana-image-masking-challenge/data?select=train.zip

标注：https://www.kaggle.com/competitions/carvana-image-masking-challenge/data?select=train_masks.zip

## 3. 准备模型与数据
您需要准备用于测试的数据集。

本例程在`scripts`目录下提供了相关模型、测试图像和测试视频的下载脚本`download.sh`，您也可以自己准备模型和数据集，并参考[4. 模型转换](#4-模型转换)进行模型转换。
```bash
chmod -R +x scripts/
./scripts/download.sh
```
执行后，模型保存至`./models`, 数据集保存至`./datasets`
```
下载的模型包括：
```
├── BM1684
│   ├── unet_fp32_1b.bmodel
│   ├── unet_int8_1b.bmodel
│   ├── unet_int8_4b.bmodel
├── BM1684X
│   ├── unet_fp32_1b.bmodel
│   └── unet_fp16_1b.bmodel
│   └── unet_int8_1b.bmodel
├── onnx
│   └── unet.onnx
└── torch
    ├── unet.pt
    └── unet_carvana_scale0.5_epoch2.pth
```

## 4. 模型转换
模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。

模型编译前需要安装TPU-NNTC(>=3.1.0)，具体可参考[tpu-nntc环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-nntc环境搭建)。

### 4.1 生成fp32 bmodel
模型编译为FP32 BModel，具体方法可参考[BMNETP 使用](https://doc.sophgo.com/docs/3.0.0/docs_latest_release/nntc/html/usage/bmnetp.html)。

本例程在`scripts`目录下提供了编译FP32 BModel的脚本。请注意修改`gen_fp32bmodel.sh`中的模型路径、生成模型目录和输入大小shapes等参数.

```bash
./scripts/gen_fp32bmodel.sh
```
执行上述命令会在`./models/BM1684`及`./models/BM1684X`下生成`unet_fp32_b1`文件，即转换好的FP32 BModel。

## 5. 例程测试
* [C++例程](./cpp/README.md)
* [Python例程](./python/README.md)

## 6. 精度测试
### 6.1 测试方法
首先，参考[C++例程](cpp/README.md)或[Python例程](python/README.md)推理要测试的数据集，保存预测结果，注意修改修改相关参数(out_threshold=0.5、n_classes=2)。 

然后，使用`tools`目录下的`eval_carvana.py`脚本，将保存的图像文件与标签文件进行对比，计算出分割任务的评价指标，命令如下：
```bash
# 请根据实际情况修改图像文件路径
python3 tools/eval_carvana.py --pred_path cpp/unet_bmcv/results/images --label_path datasets/label
python3 tools/eval_carvana.py --pred_path python/results/images --label_path datasets/label
```
### 6.2 测试结果
|   测试平台    |      测试程序        |                     测试模型                   |  mIoU | aAcc  |
| ------------ | ------------------- | ---------------------------------------------- | ----- |-------|
| BM1684 PCIe  | unet_opencv.py | unet_fp32_1b.bmodel | 98.23 | 99.25 |
| BM1684 PCIe  | unet_opencv.py | unet_int8_1b.bmodel | 97.88 | 99.30 |
| BM1684 PCIe  | unet_opencv.py | unet_int8_4b.bmodel | 97.88 | 99.17 |
| BM1684 PCIe  | unet_bmcv.soc | unet_fp32_1b.bmodel | 98.23 | 98.98 |
| BM1684 PCIe  | unet_bmcv.soc | unet_int8_1b.bmodel | 97.99 | 99.02 |
| BM1684 PCIe  | unet_bmcv.soc | unet_int8_4b.bmodel | 97.96 | 99.17 |
| BM1684X PCIe  | unet_opencv.py | unet_fp32_1b.bmodel | 98.23 | 98.98 |
| BM1684X PCIe  | unet_opencv.py | unet_fp16_1b.bmodel | 98.22 | 98.98 |
| BM1684X PCIe  | unet_opencv.py | unet_int8_1b.bmodel | 98.23  | 98.98 |
| BM1684X PCIe  | unet_bmcv.soc | unet_fp32_1b.bmodel | 98.23 | 98.98 |
| BM1684X PCIe  | unet_bmcv.soc | unet_fp16_1b.bmodel | 98.22 | 98.98 |
| BM1684X PCIe  | unet_bmcv.soc | unet_int8_1b.bmodel | 98.23  | 98.98 |

## 7. 性能测试
### 7.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684/unet_fp32_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

|                  测试模型                   | calculate time(ms) |
| ------------------------------------------- | ----------------- |
| BM1684/unet_fp32_1b.bmodel  | 487.71              |
| BM1684/unet_int8_1b.bmodel  | 307.15              |
| BM1684/unet_int8_4b.bmodel  | 218.70              |
| BM1684X/unet_fp32_1b.bmodel | 1085.02              |
| BM1684X/unet_fp16_1b.bmodel | 102.13               |
| BM1684X/unet_int8_1b.bmodel | 56.57               |

> **测试说明**：  
> 1. 性能测试结果具有一定的波动性；
> 2. `calculate time`已折算为平均每张图片的推理时间；
> 3. SoC和PCIe的测试结果基本一致。

### 7.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++例程打印的预处理时间、推理时间、后处理时间为整个batch处理的时间，需除以相应的batch size才是每张图片的处理时间。

在不同的测试平台上，使用不同的例程、模型测试`carvana`数据集，性能测试结果如下：
|    测试平台  |     测试程序      |             测试模型                |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ---------------- | ----------------------------------- | -------- | ---------     | ---------     | --------- |
| BM1684 SoC  | unet_opencv.py | unet_fp32_1b.bmodel | 81.56     | 11.71          | 491.12          | 1367.53       |
| BM1684 SoC  | unet_opencv.py | unet_int8_1b.bmodel | 81.95     | 11.17          | 310.57          | 1356.17       |
| BM1684 SoC  | unet_bmcv.soc | unet_fp32_1b.bmodel | 8.99     | 31.98          | 487.71          | 19.02       |
| BM1684 SoC  | unet_bmcv.soc | unet_int8_1b.bmodel | 8.97     | 31.93          | 307.18          | 18.87       |
| BM1684 SoC  | unet_bmcv.soc | unet_int8_1b.bmodel | 8.81     | 31.78          | 81.20          | 18.87       |
| BM1684X SoC  | unet_opencv.py | unet_fp32_1b.bmodel | 11.63     | 12.78          | 1088.92          | 1347.36       |
| BM1684X SoC  | unet_opencv.py | unet_fp16_1b.bmodel | 11.57     | 12.90          | 105.99          | 1346.43       |
| BM1684X SoC  | unet_opencv.py | unet_int8_1b.bmodel | 11.68     | 12.32          | 56.57          | 1380.37       |
| BM1684X SoC  | unet_bmcv.soc | unet_fp32_1b.bmodel | 6.95     | 6.78          | 1085.03          | 16.04       |
| BM1684X SoC  | unet_bmcv.soc | unet_fp16_1b.bmodel | 6.93     | 6.80          | 102.22          | 16.03       |
| BM1684X SoC  | unet_bmcv.soc | unet_int8_1b.bmodel | 6.95     | 6.82          | 52.88          | 16.02       |

> **测试说明**：  
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. BM1684/1684X SoC的主控CPU均为8核 ARM A53 42320 DMIPS @2.3GHz，PCIe上的性能由于CPU的不同可能存在较大差异；
> 4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异，不同的阈值对后处理时间影响较大。 

## 8. FAQ
其他问题请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。
