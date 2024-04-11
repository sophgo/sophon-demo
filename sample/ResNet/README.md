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
本例程对[torchvision Resnet](https://pytorch.org/vision/stable/models.html)的模型和算法进行移植，使之能在SOPHON BM1684\BM1684X\BM1688\CV186X上进行推理测试。

**论文:** [Resnet论文](https://arxiv.org/abs/1512.03385)

深度残差网络（Deep residual network, ResNet）是由于Kaiming He等在2015提出的深度神经网络结构，它利用残差学习来解决深度神经网络训练退化的问题。

在此非常感谢Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun等人的贡献。

## 2. 特性
* 支持BM1688/CV186X(SoC)、BM1684X(x86 PCIe、SoC)、BM1684(x86 PCIe、SoC、arm PCIe)
* 支持FP32、FP16(BM1688/BM1684X/CV186X)、INT8模型编译和推理
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
├── BM1688
│   ├── resnet50_fp16_1b.bmodel       # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1, num_core=2
│   ├── resnet50_fp32_1b.bmodel       # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1, num_core=2
│   ├── resnet50_int8_1b.bmodel       # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=1, num_core=2
│   ├── resnet50_int8_4b.bmodel       # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=1, num_core=2
│   ├── resnet50_fp16_1b_2core.bmodel # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1, num_core=2
│   ├── resnet50_fp32_1b_2core.bmodel # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1, num_core=2
│   ├── resnet50_int8_1b_2core.bmodel # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=1, num_core=2
│   └── resnet50_int8_4b_2core.bmodel # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=4, num_core=2
├── CV186X
│   ├── resnet50_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于CV186X的FP32 BModel，batch_size=1
│   ├── resnet50_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于CV186X的FP16 BModel，batch_size=1
│   ├── resnet50_int8_1b.bmodel   # 使用TPU-MLIR编译，用于CV186X的INT8 BModel，batch_size=1
│   └── resnet50_int8_4b.bmodel   # 使用TPU-MLIR编译，用于CV186X的INT8 BModel，batch_size=4
├── torch
│   ├── resnet50-11ad3fa6.pth                         # 原始模型
│   └── resnet50-11ad3fa6.torchscript.pt              # trace后的torchscript模型
└── onnx
    ├── resnet50_dynamic.onnx                          # 导出的动态onnx模型
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
导出的模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。建议使用TPU-MLIR编译BModel。

模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index/material/31/all.html)相应版本的SDK中获取)。

- 生成FP32 BModel

本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684/BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684 #bm1684x/bm1688/cv186x
```

执行上述命令会在`models/BM1684`等文件夹下生成`resnet50_fp32_1b.bmodel`文件，即转换好的FP32 BModel。

- 生成FP16 BModel

本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x #bm1688/cv186x
```

执行上述命令会在`models/BM1684X/`等文件夹下生成`resnet50_fp16_1b.bmodel`文件，即转换好的FP16 BModel。

- 生成INT8 BModel

本例程在`scripts`目录下提供了量化INT8 BModel的脚本，请注意修改`gen_int8bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台（**支持BM1684/BM1684X/BM1688/CV186X**），如：

```shell
./scripts/gen_int8bmodel_mlir.sh bm1684 #bm1684x/bm1688/cv186x
```

上述脚本会在`models/BM1684`等文件夹下生成`resnet50_int8_1b.bmodel`等文件，即转换好的INT8 BModel。

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
|   测试平台   |      测试程序      |        测试模型        | ACC(%) |
| ------------ | ----------------   | ---------------------- | ------ |
| SE5-16       | resnet_opencv.py   | resnet50_fp32_1b.bmodel   |    80.10 |
| SE5-16       | resnet_opencv.py   | resnet50_int8_1b.bmodel   |    78.70 |
| SE5-16       | resnet_opencv.py   | resnet50_int8_4b.bmodel   |    78.70 |
| SE5-16       | resnet_bmcv.py     | resnet50_fp32_1b.bmodel   |    79.90 |
| SE5-16       | resnet_bmcv.py     | resnet50_int8_1b.bmodel   |    78.50 |
| SE5-16       | resnet_bmcv.py     | resnet50_int8_4b.bmodel   |    78.50 |
| SE5-16       | resnet_opencv.soc  | resnet50_fp32_1b.bmodel   |    80.20 |
| SE5-16       | resnet_opencv.soc  | resnet50_int8_1b.bmodel   |    78.20 |
| SE5-16       | resnet_opencv.soc  | resnet50_int8_4b.bmodel   |    78.20 |
| SE5-16       | resnet_bmcv.soc    | resnet50_fp32_1b.bmodel   |    79.90 |
| SE5-16       | resnet_bmcv.soc    | resnet50_int8_1b.bmodel   |    78.50 |
| SE5-16       | resnet_bmcv.soc    | resnet50_int8_4b.bmodel   |    78.50 |
| SE7-32       | resnet_opencv.py   | resnet50_fp32_1b.bmodel   |    80.10 |
| SE7-32       | resnet_opencv.py   | resnet50_fp16_1b.bmodel   |    80.10 |
| SE7-32       | resnet_opencv.py   | resnet50_int8_1b.bmodel   |    79.10 |
| SE7-32       | resnet_opencv.py   | resnet50_int8_4b.bmodel   |    79.10 |
| SE7-32       | resnet_bmcv.py     | resnet50_fp32_1b.bmodel   |    80.00 |
| SE7-32       | resnet_bmcv.py     | resnet50_fp16_1b.bmodel   |    80.00 |
| SE7-32       | resnet_bmcv.py     | resnet50_int8_1b.bmodel   |    79.40 |
| SE7-32       | resnet_bmcv.py     | resnet50_int8_4b.bmodel   |    79.40 |
| SE7-32       | resnet_opencv.soc  | resnet50_fp32_1b.bmodel   |    80.00 |
| SE7-32       | resnet_opencv.soc  | resnet50_fp16_1b.bmodel   |    80.00 |
| SE7-32       | resnet_opencv.soc  | resnet50_int8_1b.bmodel   |    79.20 |
| SE7-32       | resnet_opencv.soc  | resnet50_int8_4b.bmodel   |    79.20 |
| SE7-32       | resnet_bmcv.soc    | resnet50_fp32_1b.bmodel   |    80.00 |
| SE7-32       | resnet_bmcv.soc    | resnet50_fp16_1b.bmodel   |    80.00 |
| SE7-32       | resnet_bmcv.soc    | resnet50_int8_1b.bmodel   |    79.40 |
| SE7-32       | resnet_bmcv.soc    | resnet50_int8_4b.bmodel   |    79.40 |
| SE9-16       | resnet_opencv.py   | resnet50_fp32_1b.bmodel   |    80.10 |
| SE9-16       | resnet_opencv.py   | resnet50_fp16_1b.bmodel   |    80.10 |
| SE9-16       | resnet_opencv.py   | resnet50_int8_1b.bmodel   |    79.90 |
| SE9-16       | resnet_opencv.py   | resnet50_int8_4b.bmodel   |    79.90 |
| SE9-16       | resnet_bmcv.py     | resnet50_fp32_1b.bmodel   |    80.00 |
| SE9-16       | resnet_bmcv.py     | resnet50_fp16_1b.bmodel   |    80.00 |
| SE9-16       | resnet_bmcv.py     | resnet50_int8_1b.bmodel   |    80.50 |
| SE9-16       | resnet_bmcv.py     | resnet50_int8_4b.bmodel   |    80.50 |
| SE9-16       | resnet_opencv.soc  | resnet50_fp32_1b.bmodel   |    80.30 |
| SE9-16       | resnet_opencv.soc  | resnet50_fp16_1b.bmodel   |    80.30 |
| SE9-16       | resnet_opencv.soc  | resnet50_int8_1b.bmodel   |    80.20 |
| SE9-16       | resnet_opencv.soc  | resnet50_int8_4b.bmodel   |    80.20 |
| SE9-16       | resnet_bmcv.soc    | resnet50_fp32_1b.bmodel   |    80.00 |
| SE9-16       | resnet_bmcv.soc    | resnet50_fp16_1b.bmodel   |    80.00 |
| SE9-16       | resnet_bmcv.soc    | resnet50_int8_1b.bmodel   |    80.50 |
| SE9-16       | resnet_bmcv.soc    | resnet50_int8_4b.bmodel   |    80.50 |
| SE9-16       | resnet_opencv.py   | resnet50_fp32_1b_2core.bmodel |    80.10 |
| SE9-16       | resnet_opencv.py   | resnet50_fp16_1b_2core.bmodel |    80.10 |
| SE9-16       | resnet_opencv.py   | resnet50_int8_1b_2core.bmodel |    79.90 |
| SE9-16       | resnet_opencv.py   | resnet50_int8_4b_2core.bmodel |    79.90 |
| SE9-16       | resnet_bmcv.py     | resnet50_fp32_1b_2core.bmodel |    80.00 |
| SE9-16       | resnet_bmcv.py     | resnet50_fp16_1b_2core.bmodel |    80.00 |
| SE9-16       | resnet_bmcv.py     | resnet50_int8_1b_2core.bmodel |    80.50 |
| SE9-16       | resnet_bmcv.py     | resnet50_int8_4b_2core.bmodel |    80.50 |
| SE9-16       | resnet_opencv.soc  | resnet50_fp32_1b_2core.bmodel |    80.30 |
| SE9-16       | resnet_opencv.soc  | resnet50_fp16_1b_2core.bmodel |    80.30 |
| SE9-16       | resnet_opencv.soc  | resnet50_int8_1b_2core.bmodel |    80.20 |
| SE9-16       | resnet_opencv.soc  | resnet50_int8_4b_2core.bmodel |    80.20 |
| SE9-16       | resnet_bmcv.soc    | resnet50_fp32_1b_2core.bmodel |    80.00 |
| SE9-16       | resnet_bmcv.soc    | resnet50_fp16_1b_2core.bmodel |    80.00 |
| SE9-16       | resnet_bmcv.soc    | resnet50_int8_1b_2core.bmodel |    80.50 |
| SE9-16       | resnet_bmcv.soc    | resnet50_int8_4b_2core.bmodel |    80.50 |
| SE9-8        | resnet_opencv.py   | resnet50_fp32_1b.bmodel  | 80.10  |
| SE9-8        | resnet_opencv.py   | resnet50_fp16_1b.bmodel  | 80.10  |
| SE9-8        | resnet_opencv.py   | resnet50_int8_1b.bmodel  | 79.90  |
| SE9-8        | resnet_opencv.py   | resnet50_int8_4b.bmodel  | 79.90  |
| SE9-8        | resnet_bmcv.py     | resnet50_fp32_1b.bmodel  | 80.00  |
| SE9-8        | resnet_bmcv.py     | resnet50_fp16_1b.bmodel  | 80.00  |
| SE9-8        | resnet_bmcv.py     | resnet50_int8_1b.bmodel  | 80.50  |
| SE9-8        | resnet_bmcv.py     | resnet50_int8_4b.bmodel  | 80.50  |
| SE9-8        | resnet_opencv.soc  | resnet50_fp32_1b.bmodel  | 80.30  |
| SE9-8        | resnet_opencv.soc  | resnet50_fp16_1b.bmodel  | 80.30  |
| SE9-8        | resnet_opencv.soc  | resnet50_int8_1b.bmodel  | 80.20  |
| SE9-8        | resnet_opencv.soc  | resnet50_int8_4b.bmodel  | 80.20  |
| SE9-8        | resnet_bmcv.soc    | resnet50_fp32_1b.bmodel  | 80.00  |
| SE9-8        | resnet_bmcv.soc    | resnet50_fp16_1b.bmodel  | 80.00  |
| SE9-8        | resnet_bmcv.soc    | resnet50_int8_1b.bmodel  | 80.50  |
| SE9-8        | resnet_bmcv.soc    | resnet50_int8_4b.bmodel  | 80.50  |

> **测试说明**：  
> 1. 由于sdk版本之间可能存在差异，实际运行结果与本表有<1%的精度误差是正常的；
> 2. 在搭载了相同TPU和SOPHONSDK的PCIe或SoC平台上，相同程序的精度一致，SE5系列对应BM1684，SE7系列对应BM1684X，SE9系列中，SE9-16对应BM1688，SE9-8对应CV186X；

## 7. 性能测试
### 7.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684/resnet50_fp32_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

|                  测试模型              | calculate time(ms) |
| -----------------------------         | ----------------- |
| BM1684/resnet50_fp32_1b.bmodel     |           6.36  |
| BM1684/resnet50_int8_1b.bmodel     |           3.93  |
| BM1684/resnet50_int8_4b.bmodel     |           1.24  |
| BM1684X/resnet50_fp32_1b.bmodel    |           9.14  |
| BM1684X/resnet50_fp16_1b.bmodel    |           1.62  |
| BM1684X/resnet50_int8_1b.bmodel    |           1.10  |
| BM1684X/resnet50_int8_4b.bmodel    |           0.82  |
| BM1688/resnet50_fp32_1b.bmodel     |          46.90  |
| BM1688/resnet50_fp16_1b.bmodel     |           8.26  |
| BM1688/resnet50_int8_1b.bmodel     |           3.14  |
| BM1688/resnet50_int8_4b.bmodel     |           2.48  |
| BM1688/resnet50_fp32_1b_2core.bmodel|          34.87  |
| BM1688/resnet50_fp16_1b_2core.bmodel|           7.55  |
| BM1688/resnet50_int8_1b_2core.bmodel|           3.03  |
| BM1688/resnet50_int8_4b_2core.bmodel|           1.92  |
| CV186X/resnet50_fp32_1b.bmodel      |          42.90  |
| CV186X/resnet50_fp16_1b.bmodel      |          6.89   |
| CV186X/resnet50_int8_1b.bmodel      |          2.43   |
| CV186X/resnet50_int8_4b.bmodel      |          1.82   |

> **测试说明**：  
> 1. 性能测试结果具有一定的波动性；
> 2. `calculate time`已折算为平均每张图片的推理时间；
> 3. SoC和PCIe的测试结果基本一致。

### 7.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++和Python例程打印的时间已经折算为单张图片的处理时间。

在不同的测试平台上，使用不同的例程、模型测试`datasets/imagenet_val_1k`，性能测试结果如下：
|    测试平台  |     测试程序        |        测试模型       |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ------------------ | --------------------- | -------- | --------- | --------- | --------- |
|   SE5-16    | resnet_opencv.py  |      resnet50_fp32_1b.bmodel      |      10.95      |      7.70       |      8.86       |      0.30       |
|   SE5-16    | resnet_opencv.py  |      resnet50_int8_1b.bmodel      |      10.12      |      7.66       |      6.39       |      0.30       |
|   SE5-16    | resnet_opencv.py  |      resnet50_int8_4b.bmodel      |      10.14      |      7.74       |      3.19       |      0.11       |
|   SE5-16    |  resnet_bmcv.py   |      resnet50_fp32_1b.bmodel      |      1.71       |      0.96       |      6.84       |      0.25       |
|   SE5-16    |  resnet_bmcv.py   |      resnet50_int8_1b.bmodel      |      1.71       |      0.97       |      4.42       |      0.26       |
|   SE5-16    |  resnet_bmcv.py   |      resnet50_int8_4b.bmodel      |      1.49       |      0.85       |      1.37       |      0.10       |
|   SE5-16    | resnet_opencv.soc |      resnet50_fp32_1b.bmodel      |      1.37       |      5.83       |      6.33       |      0.09       |
|   SE5-16    | resnet_opencv.soc |      resnet50_int8_1b.bmodel      |      1.37       |      5.86       |      3.92       |      0.09       |
|   SE5-16    | resnet_opencv.soc |      resnet50_int8_4b.bmodel      |      1.16       |      5.91       |      1.23       |      0.07       |
|   SE5-16    |  resnet_bmcv.soc  |      resnet50_fp32_1b.bmodel      |      2.54       |      2.51       |      6.31       |      0.11       |
|   SE5-16    |  resnet_bmcv.soc  |      resnet50_int8_1b.bmodel      |      2.48       |      2.50       |      3.89       |      0.11       |
|   SE5-16    |  resnet_bmcv.soc  |      resnet50_int8_4b.bmodel      |      2.44       |      2.44       |      1.22       |      0.10       |
|   SE7-32    | resnet_opencv.py  |      resnet50_fp32_1b.bmodel      |      10.12      |      7.63       |      11.83      |      0.30       |
|   SE7-32    | resnet_opencv.py  |      resnet50_fp16_1b.bmodel      |      10.10      |      7.63       |      4.31       |      0.31       |
|   SE7-32    | resnet_opencv.py  |      resnet50_int8_1b.bmodel      |      10.09      |      7.60       |      3.77       |      0.30       |
|   SE7-32    | resnet_opencv.py  |      resnet50_int8_4b.bmodel      |      9.97       |      7.64       |      3.07       |      0.11       |
|   SE7-32    |  resnet_bmcv.py   |      resnet50_fp32_1b.bmodel      |      1.52       |      0.72       |      9.68       |      0.26       |
|   SE7-32    |  resnet_bmcv.py   |      resnet50_fp16_1b.bmodel      |      1.52       |      0.72       |      2.16       |      0.26       |
|   SE7-32    |  resnet_bmcv.py   |      resnet50_int8_1b.bmodel      |      1.52       |      0.73       |      1.61       |      0.26       |
|   SE7-32    |  resnet_bmcv.py   |      resnet50_int8_4b.bmodel      |      1.32       |      0.62       |      0.96       |      0.10       |
|   SE7-32    | resnet_opencv.soc |      resnet50_fp32_1b.bmodel      |      1.15       |      5.67       |      9.12       |      0.09       |
|   SE7-32    | resnet_opencv.soc |      resnet50_fp16_1b.bmodel      |      1.17       |      5.69       |      1.64       |      0.09       |
|   SE7-32    | resnet_opencv.soc |      resnet50_int8_1b.bmodel      |      1.16       |      5.68       |      1.09       |      0.09       |
|   SE7-32    | resnet_opencv.soc |      resnet50_int8_4b.bmodel      |      0.99       |      5.75       |      0.81       |      0.07       |
|   SE7-32    |  resnet_bmcv.soc  |      resnet50_fp32_1b.bmodel      |      2.18       |      0.45       |      9.12       |      0.11       |
|   SE7-32    |  resnet_bmcv.soc  |      resnet50_fp16_1b.bmodel      |      2.16       |      0.45       |      1.61       |      0.11       |
|   SE7-32    |  resnet_bmcv.soc  |      resnet50_int8_1b.bmodel      |      2.19       |      0.45       |      1.08       |      0.11       |
|   SE7-32    |  resnet_bmcv.soc  |      resnet50_int8_4b.bmodel      |      2.11       |      0.41       |      0.81       |      0.10       |
|   SE9-16    | resnet_opencv.py  |      resnet50_fp32_1b.bmodel      |      13.95      |      10.68      |      49.07      |      0.42       |
|   SE9-16    | resnet_opencv.py  |      resnet50_fp16_1b.bmodel      |      13.02      |      10.65      |      10.93      |      0.43       |
|   SE9-16    | resnet_opencv.py  |      resnet50_int8_1b.bmodel      |      12.94      |      10.64      |      6.13       |      0.42       |
|   SE9-16    | resnet_opencv.py  |      resnet50_int8_4b.bmodel      |      12.80      |      10.68      |      4.86       |      0.15       |
|   SE9-16    |  resnet_bmcv.py   |      resnet50_fp32_1b.bmodel      |      3.17       |      1.71       |      46.33      |      0.38       |
|   SE9-16    |  resnet_bmcv.py   |      resnet50_fp16_1b.bmodel      |      3.07       |      1.71       |      8.17       |      0.38       |
|   SE9-16    |  resnet_bmcv.py   |      resnet50_int8_1b.bmodel      |      3.07       |      1.71       |      3.38       |      0.37       |
|   SE9-16    |  resnet_bmcv.py   |      resnet50_int8_4b.bmodel      |      2.88       |      1.50       |      2.27       |      0.14       |
|   SE9-16    | resnet_opencv.soc |      resnet50_fp32_1b.bmodel      |      2.46       |      7.65       |      45.40      |      0.14       |
|   SE9-16    | resnet_opencv.soc |      resnet50_fp16_1b.bmodel      |      2.42       |      7.60       |      7.36       |      0.13       |
|   SE9-16    | resnet_opencv.soc |      resnet50_int8_1b.bmodel      |      2.43       |      7.65       |      2.57       |      0.13       |
|   SE9-16    | resnet_opencv.soc |      resnet50_int8_4b.bmodel      |      2.11       |      7.74       |      2.06       |      0.10       |
|   SE9-16    |  resnet_bmcv.soc  |      resnet50_fp32_1b.bmodel      |      4.11       |      1.31       |      45.40      |      0.19       |
|   SE9-16    |  resnet_bmcv.soc  |      resnet50_fp16_1b.bmodel      |      3.96       |      1.29       |      7.37       |      0.17       |
|   SE9-16    |  resnet_bmcv.soc  |      resnet50_int8_1b.bmodel      |      3.91       |      1.30       |      2.56       |      0.16       |
|   SE9-16    |  resnet_bmcv.soc  |      resnet50_int8_4b.bmodel      |      3.83       |      1.17       |      2.06       |      0.14       |
|   SE9-16    | resnet_opencv.py  |   resnet50_fp32_1b_2core.bmodel   |      12.97      |      10.66      |      36.99      |      0.42       |
|   SE9-16    | resnet_opencv.py  |   resnet50_fp16_1b_2core.bmodel   |      12.98      |      10.73      |      10.19      |      0.42       |
|   SE9-16    | resnet_opencv.py  |   resnet50_int8_1b_2core.bmodel   |      12.90      |      10.64      |      6.02       |      0.42       |
|   SE9-16    | resnet_opencv.py  |   resnet50_int8_4b_2core.bmodel   |      12.85      |      10.63      |      4.25       |      0.15       |
|   SE9-16    |  resnet_bmcv.py   |   resnet50_fp32_1b_2core.bmodel   |      3.14       |      1.72       |      34.25      |      0.38       |
|   SE9-16    |  resnet_bmcv.py   |   resnet50_fp16_1b_2core.bmodel   |      3.10       |      1.71       |      7.48       |      0.37       |
|   SE9-16    |  resnet_bmcv.py   |   resnet50_int8_1b_2core.bmodel   |      3.18       |      1.73       |      3.29       |      0.38       |
|   SE9-16    |  resnet_bmcv.py   |   resnet50_int8_4b_2core.bmodel   |      2.79       |      1.50       |      1.69       |      0.14       |
|   SE9-16    | resnet_opencv.soc |   resnet50_fp32_1b_2core.bmodel   |      2.46       |      7.60       |      33.40      |      0.14       |
|   SE9-16    | resnet_opencv.soc |   resnet50_fp16_1b_2core.bmodel   |      2.47       |      7.66       |      6.66       |      0.14       |
|   SE9-16    | resnet_opencv.soc |   resnet50_int8_1b_2core.bmodel   |      2.46       |      7.67       |      2.46       |      0.14       |
|   SE9-16    | resnet_opencv.soc |   resnet50_int8_4b_2core.bmodel   |      2.09       |      7.73       |      1.48       |      0.10       |
|   SE9-16    |  resnet_bmcv.soc  |   resnet50_fp32_1b_2core.bmodel   |      4.02       |      1.29       |      33.39      |      0.17       |
|   SE9-16    |  resnet_bmcv.soc  |   resnet50_fp16_1b_2core.bmodel   |      3.97       |      1.31       |      6.64       |      0.17       |
|   SE9-16    |  resnet_bmcv.soc  |   resnet50_int8_1b_2core.bmodel   |      3.99       |      1.29       |      2.46       |      0.17       |
|   SE9-16    |  resnet_bmcv.soc  |   resnet50_int8_4b_2core.bmodel   |      3.80       |      1.20       |      1.48       |      0.14       |
|    SE9-8    | resnet_opencv.py  |      resnet50_fp32_1b.bmodel      |      13.82      |      10.75      |      46.28      |      0.43       |
|    SE9-8    | resnet_opencv.py  |      resnet50_fp16_1b.bmodel      |      12.90      |      10.69      |      10.31      |      0.43       |
|    SE9-8    | resnet_opencv.py  |      resnet50_int8_1b.bmodel      |      12.84      |      10.66      |      5.83       |      0.43       |
|    SE9-8    | resnet_opencv.py  |      resnet50_int8_4b.bmodel      |      12.74      |      10.70      |      4.58       |      0.15       |
|    SE9-8    |  resnet_bmcv.py   |      resnet50_fp32_1b.bmodel      |      2.89       |      1.61       |      43.54      |      0.38       |
|    SE9-8    |  resnet_bmcv.py   |      resnet50_fp16_1b.bmodel      |      2.87       |      1.61       |      7.52       |      0.37       |
|    SE9-8    |  resnet_bmcv.py   |      resnet50_int8_1b.bmodel      |      2.84       |      1.60       |      3.05       |      0.38       |
|    SE9-8    |  resnet_bmcv.py   |      resnet50_int8_4b.bmodel      |      2.56       |      1.40       |      1.97       |      0.14       |
|    SE9-8    | resnet_opencv.soc |      resnet50_fp32_1b.bmodel      |      2.27       |      7.22       |      42.71      |      0.13       |
|    SE9-8    | resnet_opencv.soc |      resnet50_fp16_1b.bmodel      |      2.27       |      7.26       |      6.72       |      0.13       |
|    SE9-8    | resnet_opencv.soc |      resnet50_int8_1b.bmodel      |      2.25       |      7.29       |      2.27       |      0.13       |
|    SE9-8    | resnet_opencv.soc |      resnet50_int8_4b.bmodel      |      1.88       |      7.31       |      1.77       |      0.10       |
|    SE9-8    |  resnet_bmcv.soc  |      resnet50_fp32_1b.bmodel      |      3.79       |      1.25       |      42.71      |      0.16       |
|    SE9-8    |  resnet_bmcv.soc  |      resnet50_fp16_1b.bmodel      |      3.68       |      1.25       |      6.71       |      0.16       |
|    SE9-8    |  resnet_bmcv.soc  |      resnet50_int8_1b.bmodel      |      3.68       |      1.26       |      2.26       |      0.16       |
|    SE9-8    |  resnet_bmcv.soc  |      resnet50_int8_4b.bmodel      |      3.61       |      1.17       |      1.76       |      0.14       |


> **测试说明**：  
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. 后处理只有argmax，可以忽略；

## 8. FAQ
请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。
