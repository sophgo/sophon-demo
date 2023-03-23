# YOLOv8

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
​YOLOv8是YOLO系列的的一个重大更新版本，它抛弃了以往的YOLO系类模型使用的Anchor-Base，采用了Anchor-Free的思想。YOLOv8建立在YOLO系列成功的基础上，通过对网络结构的改造，进一步提升其性能和灵活性。本例程对[​YOLOv8官方开源仓库](https://github.com/ultralytics/ultralytics)的模型和算法进行移植，使之能在SOPHON BM1684和BM1684X上进行推理测试。

## 2. 特性
* 支持BM1684X(x86 PCIe、SoC)和BM1684(x86 PCIe、SoC、arm PCIe)
* 支持FP32、FP16(BM1684X)、INT8模型编译和推理
* 支持基于BMCV预处理的C++推理
* 支持基于OpenCV和BMCV预处理的Python推理
* 支持单batch和多batch模型推理
* 支持1个输出模型推理
* 支持图片和视频测试

## 3. 准备模型与数据
如果您使用BM1684芯片，建议使用TPU-NNTC编译BModel，Pytorch模型在编译前要导出成torchscript模型或onnx模型；如果您使用BM1684X芯片，建议使用TPU-MLIR编译BModel，在使用TPU-MLIR编译前需要导出ONNX模型。具体可参考[YOLOv8模型导出](./docs/YOLOv8_Export_Guide.md)。

​同时，您需要准备用于测试的数据集，如果量化模型，还要准备用于量化的数据集。

​本例程在`scripts`目录下提供了相关模型和数据的下载脚本`download.sh`，您也可以自己准备模型和数据集，并参考[4. 模型编译](#4-模型编译)进行模型转换。

```bash
# 安装unzip，若已安装请跳过，非ubuntu系统视情况使用yum或其他方式安装
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```

执行后，模型保存至`models/`，测试数据集下载并解压至`datasets/test/`，精度测试数据集下载并解压至`datasets/coco/val2017_1000/`，量化数据集下载并解压至`datasets/coco128/`

```
下载的模型包括：
./models
├── BM1684
│   ├── yolov8s_fp32_1b.bmodel   # 使用TPU-NNTC编译，用于BM1684的FP32 BModel，batch_size=1
│   ├── yolov8s_int8_1b.bmodel   # 使用TPU-NNTC编译，用于BM1684的INT8 BModel，batch_size=1
│   └── yolov8s_int8_4b.bmodel   # 使用TPU-NNTC编译，用于BM1684的INT8 BModel，batch_size=4
├── BM1684X
│   ├── yolov8s_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=1
│   ├── yolov8s_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=1
│   ├── yolov8s_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=1
│   └── yolov8s_int8_4b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=4
│── torch
│   └── yolov8s.torchscript.pt   # trace后的torchscript模型
└── onnx
    ├── yolov8s_1b.onnx      # 导出的静态onnx模型，batch_size=1
    ├── yolov8s_4b.onnx      # 导出的静态onnx模型，batch_size=4
    └── yolov8s_qtable       # TPU-MLIR编译时，用于BM1684X的INT8 BModel混合精度量化
    
         
```
下载的数据包括：
```
./datasets
├── test                                      # 测试图片
├── test_car_person_1080P.mp4                 # 测试视频
├── coco.names                                # coco类别名文件
├── coco128                                   # coco128数据集，用于模型量化
└── coco                                      
    ├── val2017_1000                               # coco val2017_1000数据集：coco val2017中随机抽取的1000张样本
    └── instances_val2017_1000.json                # coco val2017_1000数据集关键点标签文件，用于计算精度评价指标 
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

​执行上述命令会在`models/BM1684/`下生成`yolov8s_fp32_1b.bmodel`文件，即转换好的FP32 BModel。

- 生成INT8 BModel

使用TPU-NNTC量化torchscript模型的方法可参考《TPU-NNTC开发参考手册》的“模型量化”(请从[算能官网](https://developer.sophgo.com/site/index/material/28/all.html)相应版本的SDK中获取)，以及[模型量化注意事项](../../docs/Calibration_Guide.md#1-注意事项)。

​本例程在`scripts`目录下提供了TPU-NNTC量化INT8 BModel的脚本，请注意修改`gen_int8bmodel_nntc.sh`中的torchscript模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台，如：

```shell
./scripts/gen_int8bmodel_nntc.sh BM1684
```

​上述脚本会在`models/BM1684`下生成`yolov8s_int8_1b.bmodel`等文件，即转换好的INT8 BModel。


### 4.2 TPU-MLIR编译BModel
模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#2-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index/material/31/all.html)相应版本的SDK中获取)。

- 生成FP32 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（支持BM1684X），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684x
```

​执行上述命令会在`models/BM1684X/`下生成`yolov8s_fp32_1b.bmodel`文件，即转换好的FP32 BModel。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（支持BM1684X），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x
```

​执行上述命令会在`models/BM1684X/`下生成`yolov8s_fp16_1b.bmodel`文件，即转换好的FP16 BModel。

- 生成INT8 BModel

​本例程在`scripts`目录下提供了量化INT8 BModel的脚本，请注意修改`gen_int8bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台（支持BM1684X），如：

```shell
./scripts/gen_int8bmodel_mlir.sh bm1684x
```

​上述脚本会在`models/BM1684X`下生成`yolov8s_int8_1b.bmodel`等文件，即转换好的INT8 BModel。

## 5. 例程测试
- [C++例程](./cpp/README.md)
- [Python例程](./python/README.md)

## 6. 精度测试
### 6.1 测试方法

首先，参考[C++例程](cpp/README.md#32-测试图片)或[Python例程](python/README.md#22-测试图片)推理要测试的数据集，生成预测的json文件，注意修改数据集(datasets/coco/val2017_1000)和相关参数(conf_thresh=0.001、nms_thresh=0.7)。  
然后，使用`tools`目录下的`eval_coco.py`脚本，将测试生成的json文件与测试集标签json文件进行对比，计算出目标检测的评价指标，命令如下：
```bash
# 安装pycocotools，若已安装请跳过
pip3 install pycocotools
# 请根据实际情况修改程序路径和json文件路径
python3 tools/eval_coco.py --gt_path datasets/coco/instances_val2017_1000.json --result_json results/yolov8s_fp32_1b.bmodel_val2017_1000_opencv_python_result.json
```
### 6.2 测试结果
在coco2017 val数据集上，精度测试结果如下：
|   测试平台    |      测试程序     |      测试模型          |AP@IoU=0.5:0.95|AP@IoU=0.5|
| ------------ | ---------------- | ---------------------- | ------------- | -------- |
| BM1684 PCIe  | yolov8_opencv.py | yolov8s_fp32_1b.bmodel | 0.448   | 0.609 |
| BM1684 PCIe  | yolov8_opencv.py | yolov8s_int8_1b.bmodel | 0.438   | 0.603 |
| BM1684 PCIe  | yolov8_bmcv.py   | yolov8s_fp32_1b.bmodel | 0.440   | 0.604 |
| BM1684 PCIe  | yolov8_bmcv.py   | yolov8s_int8_1b.bmodel | 0.432   | 0.597 |
| BM1684 PCIe  | yolov8_bmcv.pcie | yolov8s_fp32_1b.bmodel | 0.448   | 0.609 |
| BM1684 PCIe  | yolov8_bmcv.pcie | yolov8s_int8_1b.bmodel | 0.437   | 0.601 |
| BM1684X PCIe | yolov8_opencv.py | yolov8s_fp32_1b.bmodel | 0.448   | 0.609 |
| BM1684X PCIe | yolov8_opencv.py | yolov8s_fp16_1b.bmodel | 0.447   | 0.609 |
| BM1684X PCIe | yolov8_opencv.py | yolov8s_int8_1b.bmodel | 0.442   | 0.605 |
| BM1684X PCIe | yolov8_bmcv.py   | yolov8s_fp32_1b.bmodel | 0.440   | 0.602 |
| BM1684X PCIe | yolov8_bmcv.py   | yolov8s_fp16_1b.bmodel | 0.440   | 0.602 |
| BM1684X PCIe | yolov8_bmcv.py   | yolov8s_int8_1b.bmodel | 0.432   | 0.594 |
| BM1684X PCIe | yolov8_bmcv.pcie | yolov8s_fp32_1b.bmodel | 0.448   | 0.610 |
| BM1684X PCIe | yolov8_bmcv.pcie | yolov8s_fp16_1b.bmodel | 0.447   | 0.609 |
| BM1684X PCIe | yolov8_bmcv.pcie | yolov8s_int8_1b.bmodel | 0.441   | 0.605 |

> **测试说明**：  
1. batch_size=4和batch_size=1的模型精度一致；
2. SoC和PCIe的模型精度一致；
3. AP@IoU=0.5:0.95为area=all对应的指标。


## 7. 性能测试
### 7.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684/yolov8s_fp32_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

|              测试模型           | calculate time(ms) |
| -------------------------------| ----------------- |
| BM1684/yolov8s_fp32_1b.bmodel  |   25.8            |
| BM1684/yolov8s_int8_1b.bmodel  |   15.2            |
| BM1684/yolov8s_int8_4b.bmodel  |   7.52            |
| BM1684X/yolov8s_fp32_1b.bmodel |   29.7            |
| BM1684X/yolov8s_fp16_1b.bmodel |   7.05            |
| BM1684X/yolov8s_int8_1b.bmodel |   3.88            |
| BM1684X/yolov8s_int8_4b.bmodel |   3.67            |

> **测试说明**：  
1. 性能测试结果具有一定的波动性；
2. `calculate time`已折算为平均每张图片的推理时间；
3. SoC和PCIe的测试结果基本一致。


### 7.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++例程打印的预处理时间、推理时间、后处理时间为整个batch处理的时间，需除以相应的batch size才是每张图片的处理时间。

在不同的测试平台上，使用不同的例程、模型测试`datasets/val2017_1000`，conf_thresh=0.25，nms_thresh=0.7，性能测试结果如下：
|    测试平台  |     测试程序      |        测试模型        |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ---------------- | ---------------------- | -------- | --------- | --------- | --------- |
| BM1684 SoC  | yolov8_opencv.py | yolov8s_fp32_1b.bmodel | 15.90 | 23.54 | 31.30 | 5.50  |
| BM1684 SoC  | yolov8_opencv.py | yolov8s_int8_1b.bmodel | 15.09 | 23.06 | 33.31 | 5.47  | 
| BM1684 SoC  | yolov8_opencv.py | yolov8s_int8_4b.bmodel | 15.18 | 25.36 | 25.39 | 5.59  |
| BM1684 SoC  | yolov8_bmcv.py   | yolov8s_fp32_1b.bmodel | 2.99  | 3.00  | 27.90 | 5.31  |
| BM1684 SoC  | yolov8_bmcv.py   | yolov8s_int8_1b.bmodel | 2.98  | 2.45  | 17.28 | 5.40  |
| BM1684 SoC  | yolov8_bmcv.py   | yolov8s_int8_4b.bmodel | 2.83  | 2.28  | 9.26  | 4.87  |
| BM1684 SoC  | yolov8_bmcv.soc  | yolov8s_fp32_1b.bmodel | 5.245 | 2.478 | 25.92 | 17.94 |
| BM1684 SoC  | yolov8_bmcv.soc  | yolov8s_int8_1b.bmodel | 4.982 | 1.680 | 15.10 | 17.58 |
| BM1684 SoC  | yolov8_bmcv.soc  | yolov8s_int8_4b.bmodel | 4.931 | 1.623 | 7.492 | 17.49 |
| BM1684X SoC | yolov8_opencv.py | yolov8s_fp32_1b.bmodel | 15.03 | 22.98 | 34.80 | 5.45  |
| BM1684X SoC | yolov8_opencv.py | yolov8s_fp16_1b.bmodel | 15.03 | 22.46 | 12.14 | 5.45  |
| BM1684X SoC | yolov8_opencv.py | yolov8s_int8_1b.bmodel | 14.99 | 22.40 | 9.18  | 5.37  |
| BM1684X SoC | yolov8_opencv.py | yolov8s_int8_4b.bmodel | 15.03 | 24.77 | 8.91  | 5.47  |
| BM1684X SoC | yolov8_bmcv.py   | yolov8s_fp32_1b.bmodel | 2.69  | 2.23  | 31.25 | 5.52  |
| BM1684X SoC | yolov8_bmcv.py   | yolov8s_fp16_1b.bmodel | 2.58  | 2.23  | 8.55  | 5.53  |
| BM1684X SoC | yolov8_bmcv.py   | yolov8s_int8_1b.bmodel | 2.67  | 2.24  | 5.65  | 5.43  |
| BM1684X SoC | yolov8_bmcv.py   | yolov8s_int8_4b.bmodel | 2.45  | 2.14  | 5.17  | 4.88  |
| BM1684X SoC | yolov8_bmcv.soc  | yolov8s_fp32_1b.bmodel | 4.324 | 0.772 | 28.97 | 17.96 |
| BM1684X SoC | yolov8_bmcv.soc  | yolov8s_fp16_1b.bmodel | 4.312 | 0.772 | 6.259 | 17.80 |
| BM1684X SoC | yolov8_bmcv.soc  | yolov8s_int8_1b.bmodel | 4.276 | 0.772 | 3.350 | 17.95 |
| BM1684X SoC | yolov8_bmcv.soc  | yolov8s_int8_4b.bmodel | 4.128 | 0.736 | 3.277 | 17.70 |

> **测试说明**：  
1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
3. BM1684/1684X SoC的主控CPU均为8核 ARM A53 42320 DMIPS @2.3GHz，PCIe上的性能由于CPU的不同可能存在较大差异；
4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异，不同的阈值对后处理时间影响较大。 


## 8. FAQ
请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。