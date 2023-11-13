[简体中文](./README.md) | [English](./README_EN.md)

# YOLOx

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
YOLOx由旷世研究提出,是基于YOLO系列的改进，引入了解耦头和Anchor-free，提高算法整体的检测性能

**论文地址** (https://arxiv.org/abs/2107.08430)

**官方源码地址** (https://github.com/Megvii-BaseDetection/YOLOX)

## 2. 特性
* 支持BM1684X(x86 PCIe、SoC)和BM1684(x86 PCIe、SoC、arm PCIe)
* 支持FP32、FP16(BM1684X)、INT8模型编译和推理
* 支持基于BMCV、sail预处理的C++推理
* 支持基于OpenCV和BMCV预处理的Python推理
* 支持单batch和多batch模型推理
* 支持图片和视频测试
 
## 3. 准备模型与数据
推荐您使用新版编译工具链TPU-MLIR编译BModel，目前直接支持的框架有ONNX、Caffe和TFLite，其他框架的模型需要转换成onnx模型。如何将其他深度学习架构的网络模型转换成onnx, 可以参考onnx官网: https://github.com/onnx/tutorials ；YOLOX模型导出为onnx的方法可参考官方工具：https://github.com/Megvii-BaseDetection/YOLOX/tree/main/demo/ONNXRuntime

旧版编译工具链TPU-NNTC更新维护较慢，不推荐您编译使用。

同时，您需要准备用于测试的数据集，如果量化模型，还要准备用于量化的数据集。

本例程在`scripts`目录下提供了相关模型和数据集的下载脚本`download.sh`，您也可以自己准备模型和数据集，并参考[4. 模型编译](#4-模型编译)进行模型转换。

```bash
# 安装unzip，若已安装请跳过，非ubuntu系统视情况使用yum或其他方式安装
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```

下载的模型包括
```
./models
├── BM1684
│   ├── yolox_s_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684的FP32 BModel，batch_size=1
│   ├── yolox_s_fp32_4b.bmodel   # 使用TPU-MLIR编译，用于BM1684的FP32 BModel，batch_size=4
│   ├── yolox_s_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684的INT8 BModel，batch_size=1
│   └── yolox_s_int8_4b.bmodel   # 使用TPU-MLIR编译，用于BM1684的INT8 BModel，batch_size=4
├── BM1684X
│   ├── yolox_s_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=1
│   ├── yolox_s_fp32_4b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=4
│   ├── yolox_s_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=1
│   ├── yolox_s_fp16_4b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=4
│   ├── yolox_s_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=1
│   └── yolox_s_int8_4b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=4
│── torch
│   └── yolox_s.pt               # trace后的torchscript模型
└── onnx
    └── yolox_s.onnx             # 导出的onnx动态模型      
    └── yolox_s.qtable           # 用于MLIR混精度移植的配置文件
```

下载的数据包括：
```
./datasets
├── test                                      # 测试图片
├── test_car_person_1080P.mp4                 # 测试视频
├── coco.names                                # coco类别名文件
├── coco128                                   # coco128数据集，用于模型量化
└── coco                                      
    ├── val2017_1000                          # coco val2017_1000数据集：coco val2017中随机抽取的1000张样本
    └── instances_val2017_1000.json           # coco val2017_1000数据集标签文件，用于计算精度评价指标  
```

## 4. 模型编译
导出的模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。建议使用TPU-MLIR编译BModel。

模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md##1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index/material/31/all.html)相应版本的SDK中获取)。

- 生成FP32 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684/BM1684X**），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684
#or
./scripts/gen_fp32bmodel_mlir.sh bm1684x
```

​执行上述命令会在`models/BM1684`或`models/BM1684X/`下生成`yolox_s_fp32_1b.bmodel`文件，即转换好的FP32 BModel。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X**），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x
```

​执行上述命令会在`models/BM1684X/`下生成`yolox_s_fp16_1b.bmodel`文件，即转换好的FP16 BModel。

- 生成INT8 BModel

​本例程在`scripts`目录下提供了量化INT8 BModel的脚本，请注意修改`gen_int8bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台（**支持BM1684/BM1684X**），如：

```shell
./scripts/gen_int8bmodel_mlir.sh bm1684
#或
./scripts/gen_int8bmodel_mlir.sh bm1684x
```

​上述脚本会在`models/BM1684`或`models/BM1684X/`下生成`yolox_s_int8_1b.bmodel`等文件，即转换好的INT8 BModel。


## 5. 例程测试
- [C++例程](./cpp/README.md)
- [Python例程](./python/README.md)

## 6. 精度测试
### 6.1 测试方法

首先，参考[C++例程](cpp/README.md#32-测试图片)或[Python例程](python/README.md#22-测试图片)推理要测试的数据集，生成预测的json文件，注意修改数据集(datasets/coco/val2017_1000)和相关参数(conf_thresh=0.001、nms_thresh=0.6)。  
然后，使用`tools`目录下的`eval_coco.py`脚本，将测试生成的json文件与测试集标签json文件进行对比，计算出目标检测的评价指标，命令如下：
```bash
# 安装pycocotools，若已安装请跳过
pip3 install pycocotools
# 请根据实际情况修改程序路径和json文件路径
python3 tools/eval_coco.py --gt_path datasets/coco/instances_val2017_1000.json --result_json results/yolox_s_fp32_1b.bmodel_val2017_1000_opencv_python_result.json
```

### 6.2 测试结果
在coco2017val_1000数据集上，精度测试结果如下：
|   测试平台    |      测试程序     |        测试模型        | AP@IoU=0.5:0.95 | AP@IoU=0.5 |
| ------------ | ---------------- | ---------------------- | --------------- | ---------- |
| BM1684 PCIe  | yolox_opencv.py  | yolox_s_fp32_1b.bmodel |      0.366      |   0.530    |
| BM1684 PCIe  | yolox_opencv.py  | yolox_s_int8_1b.bmodel |      0.335      |   0.493    |
| BM1684 PCIe  | yolox_bmcv.py    | yolox_s_fp32_1b.bmodel |      0.363      |   0.525    |
| BM1684 PCIe  | yolox_bmcv.py    | yolox_s_int8_1b.bmodel |      0.329      |   0.485    |
| BM1684 PCIe  | yolox_bmcv.pcie  | yolox_s_fp32_1b.bmodel |      0.364      |   0.534    |
| BM1684 PCIe  | yolox_bmcv.pcie  | yolox_s_int8_1b.bmodel |      0.332      |   0.498    |
| BM1684 PCIe  | yolox_sail.pcie  | yolox_s_fp32_1b.bmodel |      0.351      |   0.516    |
| BM1684 PCIe  | yolox_sail.pcie  | yolox_s_int8_1b.bmodel |      0.319      |   0.478    |
| BM1684X PCIe | yolox_opencv.py  | yolox_s_fp32_1b.bmodel |      0.366      |   0.530    |
| BM1684X PCIe | yolox_opencv.py  | yolox_s_fp16_1b.bmodel |      0.366      |   0.530    |
| BM1684X PCIe | yolox_opencv.py  | yolox_s_int8_1b.bmodel |      0.357      |   0.529    |
| BM1684X PCIe | yolox_bmcv.py    | yolox_s_fp32_1b.bmodel |      0.363      |   0.525    |
| BM1684X PCIe | yolox_bmcv.py    | yolox_s_fp16_1b.bmodel |      0.363      |   0.525    |
| BM1684X PCIe | yolox_bmcv.py    | yolox_s_int8_1b.bmodel |      0.353      |   0.524    |
| BM1684X PCIe | yolox_bmcv.pcie  | yolox_s_fp32_1b.bmodel |      0.363      |   0.534    |
| BM1684X PCIe | yolox_bmcv.pcie  | yolox_s_fp16_1b.bmodel |      0.363      |   0.534    |
| BM1684X PCIe | yolox_bmcv.pcie  | yolox_s_int8_1b.bmodel |      0.351      |   0.527    |
| BM1684X PCIe | yolox_sail.pcie  | yolox_s_fp32_1b.bmodel |      0.350      |   0.516    |
| BM1684X PCIe | yolox_sail.pcie  | yolox_s_fp16_1b.bmodel |      0.350      |   0.516    |
| BM1684X PCIe | yolox_sail.pcie  | yolox_s_int8_1b.bmodel |      0.337      |   0.506    |

> **测试说明**：  
> 1. batch_size=4和batch_size=1的模型精度一致；
> 2. SoC和PCIe的模型精度一致；
> 3. AP@IoU=0.5:0.95为area=all对应的指标。

## 7. 性能测试
### 7.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684/yolox_s_fp32_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

|          测试模型              | calculate time(ms) |
| ---- ------------------------- | ----------------- |
| BM1684/yolox_s_fp32_1b.bmodel  |       26.01       |
| BM1684/yolox_s_fp32_4b.bmodel  |       25.62       |
| BM1684/yolox_s_int8_1b.bmodel  |       16.54       |
| BM1684/yolox_s_int8_4b.bmodel  |       11.72       |
| BM1684X/yolox_s_fp32_1b.bmodel |       27.92       |
| BM1684X/yolox_s_fp32_4b.bmodel |       25.63       |
| BM1684X/yolox_s_fp16_1b.bmodel |       6.27        |
| BM1684X/yolox_s_fp16_4b.bmodel |       6.15        |
| BM1684X/yolox_s_int8_1b.bmodel |       3.86        |
| BM1684X/yolox_s_int8_4b.bmodel |       3.69        |

> **测试说明**：  
> 1. 性能测试结果具有一定的波动性；
> 2. `calculate time`已折算为平均每张图片的推理时间；
> 3. SoC和PCIe的测试结果基本一致。

### 7.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++例程打印的预处理时间、推理时间、后处理时间为整个batch处理的时间，需除以相应的batch size才是每张图片的处理时间。

在不同的测试平台上，使用不同的例程、模型测试`datasets/val2017_1000`，conf_thresh=0.5，nms_thresh=0.5，性能测试结果如下：
|    测试平台  |     测试程序      |     测试模型          |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ---------------- | --------------------- | -------- | -------------- | ---------     | --------- |
| BM1684 SoC  | yolox_opencv.py | yolox_s_fp32_1b.bmodel | 15.18     | 14.06   | 39.70     | 4.15    |
| BM1684 SoC  | yolox_opencv.py | yolox_s_int8_1b.bmodel | 15.20     | 13.86   | 43.68     | 4.07    |
| BM1684 SoC  | yolox_opencv.py | yolox_s_int8_4b.bmodel | 15.18     | 15.25   | 38.06     | 5.90    |
| BM1684 SoC  | yolox_bmcv.py   | yolox_s_fp32_1b.bmodel | 3.70     | 2.88    | 28.16     | 3.99    |
| BM1684 SoC  | yolox_bmcv.py   | yolox_s_int8_1b.bmodel | 3.50     | 2.22    | 18.75     | 3.84    |
| BM1684 SoC  | yolox_bmcv.py   | yolox_s_int8_4b.bmodel | 3.38     | 2.06    | 13.31     | 4.72    |
| BM1684 SoC  | yolox_bmcv.soc  | yolox_s_fp32_1b.bmodel | 4.52     | 1.72    | 25.78     | 2.71     |
| BM1684 SoC  | yolox_bmcv.soc  | yolox_s_int8_1b.bmodel | 4.57     | 1.78    | 16.32     | 2.73     |
| BM1684 SoC  | yolox_bmcv.soc  | yolox_s_int8_4b.bmodel | 4.57     | 1.73    | 11.58     | 2.61     |
| BM1684 SoC  | yolox_sail.soc  | yolox_s_fp32_1b.bmodel | 2.58     | 3.91    | 26.24     | 2.10     |
| BM1684 SoC  | yolox_sail.soc  | yolox_s_int8_1b.bmodel | 2.59     | 2.38    | 16.83     | 2.08     |
| BM1684 SoC  | yolox_sail.soc  | yolox_s_int8_4b.bmodel | 2.51     | 2.31    | 11.97     | 2.10     |
| BM1684X SoC | yolox_opencv.py | yolox_s_fp32_1b.bmodel | 14.96    | 13.56   | 44.02     | 4.35    |
| BM1684X SoC | yolox_opencv.py | yolox_s_fp16_1b.bmodel | 15.00    | 13.43   | 22.38     | 4.35    |
| BM1684X SoC | yolox_opencv.py | yolox_s_int8_1b.bmodel | 14.94    | 12.87   | 20.00     | 4.35    |
| BM1684X SoC | yolox_opencv.py | yolox_s_int8_4b.bmodel | 14.87    | 15.42   | 20.14     | 6.35    |
| BM1684X SoC | yolox_bmcv.py   | yolox_s_fp32_1b.bmodel | 2.97     | 2.23    | 30.17     | 4.21     |
| BM1684X SoC | yolox_bmcv.py   | yolox_s_fp16_1b.bmodel | 3.03     | 2.23    | 8.55      | 4.27     |
| BM1684X SoC | yolox_bmcv.py   | yolox_s_int8_1b.bmodel | 2.97     | 2.23    | 6.12      | 4.22     |
| BM1684X SoC | yolox_bmcv.py   | yolox_s_int8_4b.bmodel | 2.90     | 2.09    | 5.56      | 5.45     |
| BM1684X SoC | yolox_bmcv.soc  | yolox_s_fp32_1b.bmodel | 4.28     | 0.95    | 29.05     | 2.68     |
| BM1684X SoC | yolox_bmcv.soc  | yolox_s_fp16_1b.bmodel | 4.20     | 0.95    | 7.10      | 2.65     |
| BM1684X SoC | yolox_bmcv.soc  | yolox_s_int8_1b.bmodel | 4.23     | 0.95    | 4.50      | 2.65     |
| BM1684X SoC | yolox_bmcv.soc  | yolox_s_int8_4b.bmodel | 4.08     | 0.93    | 4.46      | 2.92     |
| BM1684X SoC | yolox_sail.soc  | yolox_s_fp32_1b.bmodel | 2.37     | 3.67    | 29.50     | 2.05     |
| BM1684X SoC | yolox_sail.soc  | yolox_s_fp16_1b.bmodel | 2.38     | 3.62    | 7.55      | 2.05     |
| BM1684X SoC | yolox_sail.soc  | yolox_s_int8_1b.bmodel | 2.351    | 3.63    | 4.95      | 2.04     |
| BM1684X SoC | yolox_sail.soc  | yolox_s_int8_4b.bmodel | 2.21     | 3.22    | 4.86      | 2.04     |

> **测试说明**：  
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. BM1684 SoC的测试平台为标准版SE5，BM1684X SoC的测试平台为标准版SE7
> 4. BM1684/1684X SoC的主控处理器均为8核 ARM A53 42320 DMIPS @2.3GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 5. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异，不同的阈值对后处理时间影响较大。 

## 8. FAQ
[常见问题解答](../../docs/FAQ.md)

## 9. 致谢
* 感谢 “灵耘致新” 对YOLOX的python例程的优化
