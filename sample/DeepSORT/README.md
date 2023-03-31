# DeepSORT

## 目录

* [1. 简介](#1-简介)
* [2. 特性](#2-特性)
* [3. 准备模型与数据](#3-准备模型与数据)
* [4. 模型编译](#4-模型编译)
  * [4.1 TPU-NNTC编译BModel](#41-tpu-nntc编译bmodel)
  * [4.2 TPU-MLIR编译BModel](#42-tpu-mlir编译bmodel)
* [5. 例程测试](#5-例程测试)
* [6. 精度测试](#6-精度测试)
  * [6.1 测试方法](#61-测试方法)
  * [6.2 测试结果](#62-测试结果)
* [7. 性能测试](#7-性能测试)
  * [7.1 bmrt_test](#71-bmrt_test)
  * [7.2 程序运行性能](#72-程序运行性能)
* [8. FAQ](#8-faq)
  
## 1. 简介
​本例程使用[YOLOv5](../YOLOv5/README.md)中的目标检测模型，并对[Deep Sort with PyTorch](https://github.com/ZQPei/deep_sort_pytorch)的特征提取模型和算法进行移植，使之能在SOPHON BM1684和BM1684X上进行推理测试。

## 2. 特性
* 支持BM1684X(x86 PCIe、SoC)和BM1684(x86 PCIe、SoC、arm PCIe)
* 支持FP32、FP16(BM1684X)、INT8模型编译和推理
* 支持基于BMCV预处理的C++推理
* 支持基于OpenCV预处理的Python推理
* 支持单batch和多batch模型推理
* 支持MOT格式数据集(即图片文件夹)和单视频测试
 
## 3. 准备模型与数据
本例程**目标检测模型**和**特征提取模型**，目标检测模型请参考[YOLOv5](../YOLOv5/README.md#3-准备模型与数据)，下面主要介绍特征提取模型。

如果您使用BM1684芯片，建议使用TPU-NNTC编译BModel，Pytorch模型在编译前要导出成torchscript模型或onnx模型；如果您使用BM1684X芯片，建议使用TPU-MLIR编译BModel，Pytorch模型在编译前要导出成onnx模型。`tools/extractor_transform.py`是针对[Deep Sort with PyTorch](https://github.com/ZQPei/deep_sort_pytorch)中模型的转换脚本，可以一次性导出torchscript和onnx模型。**请您根据需要修改代码**。

**注意：** 建议使用`1.8.0+cpu`的torch版本来导出torchscript模型，避免因pytorch版本导致模型编译失败。
```
python3 tools/extractor_transform.py --pth_path <your .pth weights>
```

​同时，您需要准备用于测试的数据集或视频，如果量化模型，还要准备用于量化的数据集。

​本例程在`scripts`目录下提供了相关模型和数据的下载脚本`download.sh`，您也可以自己准备模型和数据集，并参考[4. 模型编译](#4-模型编译)进行模型转换。

```bash
# 安装unzip，若已安装请跳过
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```

下载的模型包括：
```bash
./models
├── BM1684
│   ├── extractor_fp32_1b.bmodel              # 使用TPU-NNTC编译，用于BM1684的FP32 BModel，batch_size=1
│   ├── extractor_fp32_4b.bmodel              # 使用TPU-NNTC编译，用于BM1684的FP32 BModel，batch_size=4
│   ├── extractor_int8_1b.bmodel              # 使用TPU-NNTC编译，用于BM1684的INT8 BModel，batch_size=1
│   ├── extractor_int8_4b.bmodel              # 使用TPU-NNTC编译，用于BM1684的INT8 BModel，batch_size=4
│   ├── yolov5s_v6.1_3output_fp32_1b.bmodel   # 从YOLOv5例程中获取，用于BM1684的FP32 BModel，batch_size=1
│   ├── yolov5s_v6.1_3output_int8_1b.bmodel   # 从YOLOv5例程中获取，用于BM1684的INT8 BModel，batch_size=1
│   └── yolov5s_v6.1_3output_int8_4b.bmodel   # 从YOLOv5例程中获取，用于BM1684的FP32 BModel，batch_size=4
├── BM1684X
│   ├── extractor_fp16_1b.bmodel              # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=1
│   ├── extractor_fp16_4b.bmodel              # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=4
│   ├── extractor_fp32_1b.bmodel              # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=1
│   ├── extractor_fp32_4b.bmodel              # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=4
│   ├── extractor_int8_1b.bmodel              # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=1
│   ├── extractor_int8_4b.bmodel              # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=4
│   ├── yolov5s_v6.1_3output_fp16_1b.bmodel   # 从YOLOv5例程中获取，用于BM1684的FP16 BModel，batch_size=1
│   ├── yolov5s_v6.1_3output_fp32_1b.bmodel   # 从YOLOv5例程中获取，用于BM1684的FP32 BModel，batch_size=1
│   ├── yolov5s_v6.1_3output_int8_1b.bmodel   # 从YOLOv5例程中获取，用于BM1684的INT8 BModel，batch_size=1
│   └── yolov5s_v6.1_3output_int8_4b.bmodel   # 从YOLOv5例程中获取，用于BM1684的INT8 BModel，batch_size=4
├── onnx
│   └── extractor.onnx                        # 由ckpt.t7导出的onnx模型
└── torch
    └── extractor.pt                          # 由ckpt.t7导出的torchscript模型
```
下载的数据包括：
```
./datasets
├── cali_set                                  # 量化数据集
├── test_car_person_1080P.mp4                 # 测试视频
└── mot15_trainset                            # MOT15的训练集，这里用于评价指标测试。 
```

## 4. 模型编译
导出的模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。如果您使用BM1684芯片，建议使用TPU-NNTC编译BModel；如果您使用BM1684X芯片，建议使用TPU-MLIR编译BModel。

### 4.1 TPU-NNTC编译BModel
模型编译前需要安装TPU-NNTC，具体可参考[TPU-NNTC环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-nntc环境搭建)。安装好后需在TPU-NNTC环境中进入例程目录。

- 生成FP32 BModel

使用TPU-NNTC将trace后的torchscript模型编译为FP32 BModel，具体方法可参考《TPU-NNTC开发参考手册》的“BMNETP 使用”(请从[算能官网](https://developer.sophgo.com/site/index/material/28/all.html)相应版本的SDK中获取)。

​本例程在`scripts`目录下提供了TPU-NNTC编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_nntc.sh`中的torchscript模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（支持BM1684和BM1684X），如：

```bash
./scripts/gen_fp32bmodel_nntc.sh BM1684
```

​执行上述命令会在`models/BM1684/`下生成`extractor_fp32_1b.bmodel`等文件，即转换好的FP32 BModel。

- 生成INT8 BModel

使用TPU-NNTC量化torchscript模型的方法可参考《TPU-NNTC开发参考手册》的“模型量化”(请从[算能官网](https://developer.sophgo.com/site/index/material/28/all.html)相应版本的SDK中获取)，以及[模型量化注意事项](../../docs/Calibration_Guide.md#1-注意事项)。

​本例程在`scripts`目录下提供了TPU-NNTC量化INT8 BModel的脚本，请注意修改`gen_int8bmodel_nntc.sh`中的torchscript模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台，如：

```shell
./scripts/gen_int8bmodel_nntc.sh BM1684
```

​上述脚本会在`models/BM1684`下生成`extractor_int8_1b.bmodel`等文件，即转换好的INT8 BModel。

### 4.2 TPU-MLIR编译BModel
模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#2-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index/material/31/all.html)相应版本的SDK中获取)。

- 生成FP32 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（支持BM1684X），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684x
```

​执行上述命令会在`models/BM1684X/`下生成`extractor_fp32_1b.bmodel`文件，即转换好的FP32 BModel。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（支持BM1684X），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x
```

​执行上述命令会在`models/BM1684X/`下生成`extractor_fp16_1b.bmodel`文件，即转换好的FP16 BModel。

- 生成INT8 BModel

​本例程在`scripts`目录下提供了量化INT8 BModel的脚本，请注意修改`gen_int8bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台（支持BM1684X），如：

```shell
./scripts/gen_int8bmodel_mlir.sh bm1684x
```

​上述脚本会在`models/BM1684X`下生成`extractor_int8_1b.bmodel`等文件，即转换好的INT8 BModel。


## 5. 例程测试
- [C++例程](./cpp/README.md)
- [Python例程](./python/README.md)

## 6. 精度测试
### 6.1 测试方法

首先，参考[C++例程](cpp/README.md#32-测试MOT数据集)或[Python例程](python/README.md#22-测试MOT数据集)推理要测试的数据集，生成包含目标追踪结果的txt文件，注意修改数据集(datasets/mot15_trainset/ADL-Rundle-6/img1)。  
然后，使用`tools`目录下的`eval_mot15.py`脚本，将测试生成的txt文件与测试集标签txt文件进行对比，计算出目标追踪的一系列评价指标，命令如下：
```bash
# 安装motmetrics，若已安装请跳过
pip3 install motmetrics
# 请根据实际情况修改程序路径和txt文件路径
python3 tools/eval_mot15.py --gt_file datasets/mot15_trainset/ADL-Rundle-6/gt/gt.txt --ts_file python/results/mot_eval/ADL-Rundle-6_extractor_fp32_1b.bmodel.txt
```
运行结果：
```bash
MOTA = 0.43801157915751643
     num_frames      IDF1       IDP       IDR      Rcll      Prcn    GT  MT  PT  ML    FP    FN  IDsw  FM      MOTA      MOTP
acc         525  0.524889  0.544908  0.506289  0.687163  0.739579  5009  10  12   2  1212  1567    36  79  0.438012  0.218005
```
### 6.2 测试结果
这里使用目标检测模型yolov5s_v6.1_3output_int8_1b.bmodel，使用数据集ADL-Rundle-6，记录MOTA作为精度指标，精度测试结果如下：
|   测试平台    |      测试程序     |           测试模型         | MOTA |
| ------------ | ---------------- | -------------------------- | ---- |
| BM1684 PCIe  | deepsort_opencv.py | extractor_fp32_1b.bmodel | 45.7 |
| BM1684 PCIe  | deepsort_opencv.py | extractor_int8_1b.bmodel | 45.6 |
| BM1684 PCIe | deepsort_bmcv.pcie | extractor_fp32_1b.bmodel | 45.6 |
| BM1684 PCIe | deepsort_bmcv.pcie | extractor_int8_1b.bmodel | 45.7 |
| BM1684x PCIe  | deepsort_opencv.py | extractor_fp32_1b.bmodel | 43.8 |
| BM1684x PCIe  | deepsort_opencv.py | extractor_fp16_1b.bmodel | 43.8 |
| BM1684x PCIe  | deepsort_opencv.py | extractor_int8_1b.bmodel | 43.1 |
| BM1684X PCIe | deepsort_bmcv.pcie | extractor_fp32_1b.bmodel | 44.2 |
| BM1684X PCIe | deepsort_bmcv.pcie | extractor_fp16_1b.bmodel | 44.2 |
| BM1684X PCIe | deepsort_bmcv.pcie | extractor_int8_1b.bmodel | 43.7 |

> **测试说明**：  
1. batch_size=4和batch_size=1的模型精度一致；
2. SoC和PCIe的模型精度一致；

## 7. 性能测试
### 7.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径
bmrt_test --bmodel models/BM1684/extractor_fp32_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

|           测试模型             | calculate time(ms) |
| -----------------------------  | ----------------- |
| BM1684/extractor_fp32_1b.bmodel  |   2.26       |
| BM1684/extractor_fp32_4b.bmodel  |   1.25       |
| BM1684/extractor_int8_1b.bmodel  |   0.99        |
| BM1684/extractor_int8_4b.bmodel  |   0.25        |
| BM1684X/extractor_fp32_1b.bmodel |   2.08        |
| BM1684X/extractor_fp32_4b.bmodel |   1.88         |
| BM1684X/extractor_fp16_1b.bmodel |   0.56         |
| BM1684X/extractor_fp16_4b.bmodel |   0.24         |
| BM1684X/extractor_int8_1b.bmodel |   0.33        |
| BM1684X/extractor_int8_4b.bmodel |   0.14         |

> **测试说明**：  
1. 性能测试结果具有一定的波动性；
2. `calculate time`已折算为平均每张图片的推理时间。

### 7.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++例程打印的预处理时间、推理时间为整个batch处理的时间，需除以相应的batch size才是每张图片的处理时间。这里**只统计特征提取模型的时间**，解码、目标检测模型的时间请参考[YOLOV5](../YOLOv5/README.md#72-程序运行性能)。

在不同的测试平台上，使用不同的例程、模型测试`datasets/mot15_trainset/ADL-Rundle-6/img1`，性能测试结果如下：
|    测试平台  |     测试程序       |        测试模型           |preprocess_time|inference_time|postprocess_time| 
| ----------- | ----------------   |------------------------- | ------------- | ------------ |  --------- |
| BM1684 soc  | deepsort_opencv.py | extractor_fp32_1b.bmodel | 2.63  | 3.43  |  94.40  |
| BM1684 soc  | deepsort_opencv.py | extractor_fp32_4b.bmodel | 2.52  | 1.95 |  74.49  |
| BM1684 soc  | deepsort_opencv.py | extractor_int8_1b.bmodel |  2.44 | 2.08 |  75.16   |
| BM1684 soc  | deepsort_opencv.py | extractor_fp32_4b.bmodel |  2.42 | 1.09 |  61.44   |
| BM1684 soc  | deepsort_bmcv.soc | extractor_fp32_1b.bmodel | 0.16| 2.19  |  4.53 |
| BM1684 soc  | deepsort_bmcv.soc | extractor_fp32_4b.bmodel | 0.09 |1.35  | 4.59  |
| BM1684 soc  | deepsort_bmcv.soc | extractor_int8_1b.bmodel | 0.15 |0.92  | 5.02   |
| BM1684 soc  | deepsort_bmcv.soc | extractor_int8_4b.bmodel | 0.09  | 0.25 | 5.05   |
| BM1684x soc | deepsort_opencv.py | extractor_fp32_1b.bmodel | 2.14  | 3.50 | 62.09   |
| BM1684x soc | deepsort_opencv.py | extractor_fp32_4b.bmodel | 2.14  | 3.15 | 66.19   |
| BM1684x soc | deepsort_opencv.py | extractor_fp16_1b.bmodel | 2.17  | 1.19 |  59.13   |
| BM1684x soc | deepsort_opencv.py | extractor_fp16_4b.bmodel | 2.14  | 1.45 | 58.72   |
| BM1684x soc | deepsort_opencv.py | extractor_int8_1b.bmodel | 2.17  | 1.19 |  59.13  |
| BM1684x soc | deepsort_opencv.py | extractor_int8_4b.bmodel | 2.15  | 0.64  | 62.25   |
| BM1684X soc | deepsort_bmcv.soc | extractor_fp32_1b.bmodel | 0.12 | 2.65 | 5.34   |
| BM1684X soc | deepsort_bmcv.soc | extractor_fp32_4b.bmodel | 0.08 | 2.31  |5.29    |
| BM1684X soc | deepsort_bmcv.soc | extractor_fp16_1b.bmodel | 0.12 | 0.61 | 5.15    |
| BM1684X soc | deepsort_bmcv.soc | extractor_fp16_4b.bmodel | 0.08 | 0.28 | 5.31   |
| BM1684X soc | deepsort_bmcv.soc | extractor_int8_1b.bmodel | 0.12 | 0.34 | 5.41    |
| BM1684X soc | deepsort_bmcv.soc | extractor_int8_4b.bmodel | 0.08 | 0.16  | 5.43   |

> **测试说明**：  
1. 时间单位均为毫秒(ms)，preprocess_time、inference_time是特征提取模型平均每个crop的处理时间，postprocess_time是deepsort算法平均每帧的后处理时间；
2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
3. BM1684/1684X SoC的主控CPU均为8核 ARM A53 42320 DMIPS @2.3GHz，PCIe上的性能由于CPU的不同可能存在较大差异；

## 8. FAQ
请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。