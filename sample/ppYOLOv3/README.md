# ppyolov3

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

ppyolov3 作为一种经典的单阶段目标检测框架，YOLO系列的目标检测算法得到了学术界与工业界们的广泛关注。由于YOLO系列属于单阶段目标检测，因而具有较快的推理速度，能够更好的满足现实场景的需求。

## 2. 特性
* 支持BM1684X(x86 PCIe、SoC)和BM1684(x86 PCIe、SoC、arm PCIe)
* 支持FP32、FP16(BM1684X)、INT8模型编译和推理
* 支持基于BMCV预处理的C++推理
* 支持基于BMCV和opencv预处理的Python推理
* 支持图片和视频测试

## 3. 准备模型与数据
建议使用TPU-MLIR编译BModel，Pytorch模型在编译前要导出成onnx模型。具体可参考[YOLOv5模型导出](./docs/YOLOv5_Export_Guide.md)。

​同时，您需要准备用于测试的数据集，如果量化模型，还要准备用于量化的数据集。

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
├── BM1684
│   ├── ppyolov3_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684的FP32 BModel，batch_size=1
│   ├── ppyolov3_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684的INT8 BModel，batch_size=1
│   └── ppyolov3_int8_4b.bmodel   # 使用TPU-MLIR编译，用于BM1684的INT8 BModel，batch_size=4
│   
├── BM1684X
│   ├── ppyolov3_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=1
│   ├── ppyolov3_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=1
│   ├── ppyolov3_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=1
│   └── ppyolov3_int8_4b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=4
│   
└── onnx
    ├── ppyolov3.onnx              # 导出的onnx模型   
    └── ppyolov3_4b.onnx           # 导出的4batch onnx模型    
```
下载的数据包括：
```
./datasets
├── test                                      # 测试图片
├── coco.names                                # coco类别名文件
├── coco128                                   # coco128数据集，用于模型量化
└── coco                                      
    ├── val2017_1000                          # coco val2017_1000数据集
    └── instances_val2017_1000.json           # coco val2017_1000数据集标签文件，用于计算精度评价指标  
```

## 4. 模型编译
导出的模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。如果您使用BM1684芯片，建议使用TPU-NNTC编译BModel；如果您使用BM1684X芯片，建议使用TPU-MLIR编译BModel。

模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index/material/31/all.html)相应版本的SDK中获取)。

- 生成FP32 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（支持BM1684X），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684
#或
./scripts/gen_fp32bmodel_mlir.sh bm1684x
```

​执行上述命令会在`models/BM1684X/`下生成`ppyolov3_fp32_1b.bmodel`文件，即转换好的FP32 BModel。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（支持BM1684X），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x
```

​执行上述命令会在`models/BM1684X/`下生成`ppyolov3_fp16_1b.bmodel`文件，即转换好的FP16 BModel。

- 生成INT8 BModel

​本例程在`scripts`目录下提供了量化INT8 BModel的脚本，请注意修改`gen_int8bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台（支持BM1684X），如：

```shell
./scripts/gen_int8bmodel_mlir.sh bm1684
#或
./scripts/gen_int8bmodel_mlir.sh bm1684x
```

​上述脚本会在`models/BM1684X`下生成`ppyolov3_int8_1b.bmodel`等文件，即转换好的INT8 BModel。


## 5. 例程测试
- [C++例程](./cpp/README.md)
- [Python例程](./python/README.md)

## 6. 精度测试
### 6.1 测试方法

首先，参考[C++例程](cpp/README.md#32-测试图片)或[Python例程](python/README.md#22-测试图片)推理要测试的数据集，生成预测的json文件，注意修改数据集(datasets/coco/val2017_1000)和相关参数(--conf_thresh=0.001 --nms_thresh=0.6)。  
然后，使用`tools`目录下的`eval_coco.py`脚本，将测试生成的json文件与测试集标签json文件进行对比，计算出目标检测的评价指标，命令如下：
```bash
# 安装pycocotools，若已安装请跳过
pip3 install pycocotools
# 请根据实际情况修改程序路径和json文件路径
python3 tools/eval_coco.py --gt_path datasets/coco/instances_val2017_1000.json --result_json results/ppyolov3_fp32_1b.bmodel_val2017_1000_bmcv_python_result.json
```
### 6.2 测试结果
在coco2017val_1000数据集上，精度测试结果如下：
|   测试平台   |      测试程序      |         测试模型        |AP@IoU=0.5:0.95|AP@IoU=0.5|
| ------------ | ------------------ | ----------------------- | ------------- | -------- |
| BM1684 PCIe  | ppyolov3_opencv.py | ppyolov3_int8_1b.bmodel | 0.289         | 0.580    |
| BM1684 PCIe  | ppyolov3_bmcv.py   | ppyolov3_int8_1b.bmodel | 0.267         | 0.538    |
| BM1684X PCIe | ppyolov3_opencv.py | ppyolov3_fp32_1b.bmodel | 0.305         | 0.591    |
| BM1684X PCIe | ppyolov3_opencv.py | ppyolov3_fp16_1b.bmodel | 0.305         | 0.591    |
| BM1684X PCIe | ppyolov3_opencv.py | ppyolov3_int8_1b.bmodel | 0.300         | 0.586    |
| BM1684X PCIe | ppyolov3_bmcv.py   | ppyolov3_fp32_1b.bmodel | 0.289         | 0.559    |
| BM1684X PCIe | ppyolov3_bmcv.py   | ppyolov3_fp16_1b.bmodel | 0.289         | 0.559    |
| BM1684X PCIe | ppyolov3_bmcv.py   | ppyolov3_int8_1b.bmodel | 0.282         | 0.551    |
| BM1684X PCIe | ppyolov3_sail.pcie | ppyolov3_fp32_1b.bmodel | 0.281         | 0.551    |
| BM1684X PCIe | ppyolov3_sail.pcie | ppyolov3_fp16_1b.bmodel | 0.281         | 0.551    | 
| BM1684X PCIe | ppyolov3_sail.pcie | ppyolov3_int8_1b.bmodel | 0.274         | 0.542    |
| BM1684X PCIe | ppyolov3_bmcv.pcie | ppyolov3_fp32_1b.bmodel | 0.279         | 0.548    |
| BM1684X PCIe | ppyolov3_bmcv.pcie | ppyolov3_fp16_1b.bmodel | 0.278         | 0.547    |
| BM1684X PCIe | ppyolov3_bmcv.pcie | ppyolov3_int8_1b.bmodel | 0.273         | 0.544    |

> **测试说明**：  
1. SoC和PCIe的模型精度一致；
2. AP@IoU=0.5:0.95为area=all对应的指标；

## 7. 性能测试
### 7.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684/ppyolov3_fp32_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

|            测试模型             | calculate time(ms)|
| ------------------------------- | ----------------- |
| BM1684/ppyolov3_fp32_1b.bmodel  | 91.2              |
| BM1684/ppyolov3_int8_1b.bmodel  | 55.7              |
| BM1684/ppyolov3_int8_4b.bmodel  | 15.3              |
| BM1684X/ppyolov3_fp32_1b.bmodel | 187.2             |
| BM1684X/ppyolov3_fp16_1b.bmodel | 18.1              |
| BM1684X/ppyolov3_int8_1b.bmodel | 6.92              |
| BM1684X/ppyolov3_int8_4b.bmodel | 6.80              |

> **测试说明**：  
1. 性能测试结果具有一定的波动性；
2. `calculate time`已折算为平均每张图片的推理时间；
3. SoC和PCIe的测试结果基本一致。

### 7.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++例程打印的预处理时间、推理时间、后处理时间为整个batch处理的时间，需除以相应的batch size才是每张图片的处理时间。

在不同的测试平台上，使用不同的例程、模型测试`datasets/coco/val2017_1000`，性能测试结果如下：
|    测试平台 |     测试程序       |        测试模型         |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ------------------ | ----------------------- | --------- | ---------- | ----------- | ----------- |
| BM1684 SoC  | ppyolov3_opencv.py | ppyolov3_int8_1b.bmodel | 3.75      | 59.37      | 59.68       | 99.58       |
| BM1684 SoC  | ppyolov3_bmcv.py   | ppyolov3_int8_1b.bmodel | 3.55      | 2.23       | 56.66       | 107.76      |
| BM1684 SoC  | ppyolov3_bmcv.soc  | ppyolov3_int8_1b.bmodel | 4.54      | 1.82       | 51.38       | 17.72       |
| BM1684X SoC | ppyolov3_opencv.py | ppyolov3_fp32_1b.bmodel | 6.08      | 14.12      | 173.25      | 21.58       |
| BM1684X SoC | ppyolov3_opencv.py | ppyolov3_fp16_1b.bmodel | 5.31      | 13.84      | 28.99       | 15.47       |
| BM1684X SoC | ppyolov3_opencv.py | ppyolov3_int8_1b.bmodel | 5.28      | 13.43      | 19.88       | 13.32       |
| BM1684X SoC | ppyolov3_bmcv.py   | ppyolov3_fp32_1b.bmodel | 3.80      | 1.79       | 176.08      | 19.43       |
| BM1684X SoC | ppyolov3_bmcv.py   | ppyolov3_fp16_1b.bmodel | 3.96      | 1.83       | 25.00       | 17.59       |
| BM1684X SoC | ppyolov3_bmcv.py   | ppyolov3_int8_1b.bmodel | 3.89      | 1.77       | 15.65       | 14.36       |
| BM1684X SoC | ppyolov3_sail.soc  | ppyolov3_fp32_1b.bmodel | 3.93      | 1.53       | 173.80      | 12.80       |
| BM1684X SoC | ppyolov3_sail.soc  | ppyolov3_fp16_1b.bmodel | 3.89      | 1.54       | 22.61       | 12.78       |
| BM1684X SoC | ppyolov3_sail.soc  | ppyolov3_int8_1b.bmodel | 3.86      | 1.49       | 13.54       | 12.64       |
| BM1684X SoC | ppyolov3_bmcv.soc  | ppyolov3_fp32_1b.bmodel | 4.43      | 1.16       | 169.99      | 17.66       |
| BM1684X SoC | ppyolov3_bmcv.soc  | ppyolov3_fp16_1b.bmodel | 4.43      | 1.16       | 17.75       | 17.77       |
| BM1684X SoC | ppyolov3_bmcv.soc  | ppyolov3_int8_1b.bmodel | 4.38      | 1.15       | 7.36        | 17.33       |


> **测试说明**：  
1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
3. BM1684/1684X SoC的主控CPU均为8核 ARM A53 42320 DMIPS @2.3GHz，PCIe上的性能由于CPU的不同可能存在较大差异；
4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异； 

## 8. FAQ
请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。
