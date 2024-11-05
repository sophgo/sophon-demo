# Resnet

## 目录

* [1. 简介](#1-简介)
* [2. 特性](#2-特性)
* [3. 准备模型与数据](#3-准备模型与数据)
* [4. 例程测试](#4-例程测试)
* [5. 精度测试](#5-精度测试)
  * [5.1 测试方法](#51-测试方法)
  * [5.2 测试结果](#52-测试结果)
* [6. 性能测试](#6-性能测试)
  * [6.1 tpurt_test](#61-tpurt_test)
  * [6.2 程序运行性能](#62-程序运行性能)
* [7. FAQ](#7-faq)
  
## 1. 简介
本例程对[torchvision Resnet](https://pytorch.org/vision/stable/models.html)的模型和算法进行移植，使之能在SOPHON BM1690上进行推理测试。

**论文:** [Resnet论文](https://arxiv.org/abs/1512.03385)

深度残差网络（Deep residual network, ResNet）是由于Kaiming He等在2015提出的深度神经网络结构，它利用残差学习来解决深度神经网络训练退化的问题。

在此非常感谢Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun等人的贡献。

## 2. 特性
* 支持BM1690(PCIe)
* 支持FP32、FP16、INT8模型推理
* 支持基于OpenCV的Python推理
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
./models
└── BM1690
    ├── resnet50_fp16_1b.bmodel       # 使用TPU-MLIR编译，用于BM1690的FP16 BModel，batch_size=1, num_core=1
    ├── resnet50_fp32_1b.bmodel       # 使用TPU-MLIR编译，用于BM1690的FP32 BModel，batch_size=1, num_core=1
    ├── resnet50_int8_1b.bmodel       # 使用TPU-MLIR编译，用于BM1690的INT8 BModel，batch_size=1, num_core=1
    ├── resnet50_int8_4b.bmodel       # 使用TPU-MLIR编译，用于BM1690的INT8 BModel，batch_size=1, num_core=1
    ├── resnet50_fp16_1b_8core.bmodel # 使用TPU-MLIR编译，用于BM1690的FP16 BModel，batch_size=1, num_core=8
    ├── resnet50_fp32_1b_8core.bmodel # 使用TPU-MLIR编译，用于BM1690的FP32 BModel，batch_size=1, num_core=8
    ├── resnet50_int8_1b_8core.bmodel # 使用TPU-MLIR编译，用于BM1690的INT8 BModel，batch_size=1, num_core=8
    └── resnet50_int8_4b_8core.bmodel # 使用TPU-MLIR编译，用于BM1690的INT8 BModel，batch_size=4, num_core=8
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

## 4. 例程测试
* [C++例程](cpp/README.md)
* [Python例程](python/README.md)

## 5. 精度测试
### 5.1 测试方法

首先，参考[C++例程](cpp/README.md#32-测试图片)或[Python例程](python/README.md#22-测试图片)推理要测试的数据集，生成预测的json文件，注意修改相关参数。  
然后，使用`tools`目录下的`eval_imagenet.py`脚本，将预测结果文件与测试集标签文件进行对比，计算出分类准确率。具体的测试命令如下：
```bash
# 请根据实际情况修改文件路径
python3 tools/eval_imagenet.py --gt_path datasets/imagenet_val_1k/label.txt --result_json results/resnet50_int8_1b.bmodel_img_opencv_python_result.json
```
### 5.2 测试结果
在imagenet_val_1k数据集上，精度测试结果如下：
|   测试平台   |      测试程序      |        测试模型        | ACC(%) |
| ------------ | ----------------   | ---------------------- | ------ |
|     BM1690 PCIe     | resnet_opencv.py   | resnet50_fp32_1b.bmodel   |    80.10 |
|     BM1690 PCIe     | resnet_opencv.py   | resnet50_int8_1b.bmodel   |    79.70 |
|     BM1690 PCIe     | resnet_opencv.py   | resnet50_int8_4b.bmodel   |    79.70 |

> **测试说明**：  
> 1. 由于sdk版本之间可能存在差异，实际运行结果与本表有<1%的精度误差是正常的；

## 6. 性能测试
### 6.1 tpurt_test
使用tpurt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
tpu-model-rt --bmodel models/BM1690/resnet50_fp32_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

|                  测试模型              | calculate time(ms) |
| -----------------------------         | ----------------- |
| BM1690/resnet50_fp32_1b.bmodel     |           7.88  |
| BM1690/resnet50_int8_1b.bmodel     |           1.16  |
| BM1690/resnet50_int8_4b.bmodel     |           0.53  |

> **测试说明**：  
> 1. 性能测试结果具有一定的波动性；
> 2. `calculate time`已折算为平均每张图片的推理时间；

### 6.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++和Python例程打印的时间已经折算为单张图片的处理时间。

在不同的测试平台上，使用不同的例程、模型测试`datasets/imagenet_val_1k`，性能测试结果如下：
|    测试平台  |     测试程序        |        测试模型       |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ------------------ | --------------------- | -------- | --------- | --------- | --------- |
|   BM1690 PCIe   | resnet_opencv.py  |      resnet50_fp32_1b.bmodel      |      5.07      |      3.08       |      8.86       |      0.17       |
|   BM1690 PCIe   | resnet_opencv.py  |      resnet50_int8_1b.bmodel      |      4.93      |      3.03       |      6.39       |      0.16       |
|   BM1690 PCIe   | resnet_opencv.py  |      resnet50_int8_4b.bmodel      |      3.47      |      1.94       |      3.19       |      0.05       |



> **测试说明**：  
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. 后处理只有argmax，可以忽略；

## 7. FAQ
请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。
