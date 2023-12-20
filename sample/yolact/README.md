[简体中文](./README.md) | [English](./README_EN.md)

# yolact

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
yolact是一种实时的实例分割的方法。
本例程对[yolact官方开源仓库](https://github.com/dbolya/yolact)的模型和算法进行移植，使之能在SOPHON BM1684和BM1684X上进行推理测试。

## 2. 特性
* 支持BM1684X(x86 PCIe)，BM1684
* 支持FP32模型编译和推理
* 支持基于OpenCV和BMCV预处理的Python推理
* 支持单batch和多batch模型推理
* 支持图片和视频测试
 
## 3. 准备模型与数据

​本例程在`scripts`目录下提供了相关模型和数据的下载脚本`download.sh`，您也可以自己准备模型和数据集，并参考[4. 模型编译](#4-模型编译)进行模型转换。

```bash
# 安装unzip，若已安装请跳过，非ubuntu系统视情况使用yum或其他方式安装
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```

下载的模型包括：
```
├── models
│   ├── BM1684					
│   │   ├── yolact_bm1684_fp32_1b.bmodel # 使用TPU-MLIR编译，用于BM1684的FP32 BModel，batch_size=1
│   │   ├── yolact_bm1684_int8_1b.bmodel # 使用TPU-MLIR编译，用于BM1684的INT8 BModel，batch_size=1
│   │   └── yolact_bm1684_int8_4b.bmodel # 使用TPU-MLIR编译，用于BM1684的INT8 BModel，batch_size=4
│   ├── BM1684X
│   │   ├── yolact_bm1684x_fp32_1b.bmodel # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=1
│   │   ├── yolact_bm1684x_fp16_1b.bmodel # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=1
│   │   ├── yolact_bm1684x_int8_1b.bmodel # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=1
│   │   └── yolact_bm1684x_int8_4b.bmodel # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=4
│── torch
│       └── yolact_base_54_800000.trace.pt	     # trace后的torchscript模型
└── onnx
    └── yolact.onnx             	     # 导出的onnx动态模型
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

模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index/material/31/all.html)相应版本的SDK中获取)。

- 生成FP32 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684/BM1684X【FP16仅支持BM1684X】**），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684
#or
./scripts/gen_fp32bmodel_mlir.sh bm1684x
```

​执行上述命令会在`models/BM1684`或`models/BM1684X/`下生成yolact_bm1684x(或bm1684)_fp32_1b.bmodel文件，即转换好的FP32 BModel。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X**），如：

./scripts/gen_fp16bmodel_mlir.sh bm1684x

​执行上述命令会在`models/BM1684X/`下生成yolact_bm1684x_fp16_1b.bmodel文件，即转换好的FP16 BModel。

- 生成INT8 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译INT8 BModel的脚本，请注意修改`gen_int8bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684/BM1684X**），如：

```bash
./scripts/gen_int8bmodel_mlir.sh bm1684
#or
./scripts/gen_int8bmodel_mlir.sh bm1684x
```

​执行上述命令会在`models/BM1684`或`models/BM1684X/`下生成yolact_bm1684x(或bm1684)_int8_1b.bmodel文件，即转换好的INT8 BModel。(也可以在./scripts/gen_int8bmodel_mlir 中修改batch size的参数得到bs=4的int8 bmodel)

## 5. 例程测试
- [Python例程](./python/README.md)
- [C++例程](./cpp/README.md)

## 6. 精度测试
### 6.1 测试方法

首先，参考[C++例程](cpp/README.md#32-测试图片)或[Python例程](python/README.md#22-测试图片)推理要测试的数据集，生成预测的json文件，注意修改数据集(datasets/coco/val2017_1000)和相关参数(conf_thresh=0.5、nms_thresh=0.5)。  
然后，使用`tools`目录下的`eval_coco.py`脚本，将测试生成的json文件与测试集标签json文件进行对比，计算出目标检测的评价指标，命令如下：
```bash
# 安装pycocotools，若已安装请跳过
pip3 install pycocotools
# 请根据实际情况修改程序路径和json文件路径
python3 tools/eval_coco.py --gt_path datasets/coco/instances_val2017_1000.json --result_json results/yolact_bm1684_fp32_1b.bmodel_val2017_1000_opencv_python_result.json
```
### 6.2 测试结果
在coco2017val_1000数据集上，精度测试结果如下：
|   测试平台    |      测试程序      |              测试模型          |AP@IoU=0.5:0.95|AP@IoU=0.5|
| ------------ | ---------------- | ----------------------------- | ------------- | -------- |
| BM1684X PCIe | yolact_opencv.py | yolact_bm1684x_fp32_1b.bmodel | 0.262         | 0.393    |
| BM1684X PCIe | yolact_opencv.py | yolact_bm1684x_fp16_1b.bmodel | 0.262         | 0.392    |
| BM1684X PCIe | yolact_opencv.py | yolact_bm1684x_int8_1b.bmodel | 0.258         | 0.389    |
| BM1684X PCIe | yolact_bmcv.py   | yolact_bm1684x_fp32_1b.bmodel | 0.261         | 0.390    |
| BM1684X PCIe | yolact_bmcv.py   | yolact_bm1684x_fp16_1b.bmodel | 0.262         | 0.393    |
| BM1684X PCIe | yolact_bmcv.py   | yolact_bm1684x_int8_1b.bmodel | 0.259         | 0.388    |
| BM1684X PCIe | yolact_bmcv.pcie | yolact_bm1684x_fp32_1b.bmodel | 0.261         | 0.390    |
| BM1684X PCIe | yolact_bmcv.pcie | yolact_bm1684x_fp16_1b.bmodel | 0.262         | 0.392    |
| BM1684X PCIe | yolact_bmcv.pcie | yolact_bm1684x_int8_1b.bmodel | 0.260         | 0.389    |
| BM1684 PCIe  | yolact_opencv.py | yolact_bm1684_fp32_1b.bmodel  | 0.262         | 0.393    | 
| BM1684 PCIe  | yolact_opencv.py | yolact_bm1684_int8_1b.bmodel  | 0.254         | 0.386    | 
| BM1684 PCIe  | yolact_bmcv.py   | yolact_bm1684_fp32_1b.bmodel  | 0.261         | 0.391    | 
| BM1684 PCIe  | yolact_bmcv.py   | yolact_bm1684_int8_1b.bmodel  | 0.252         | 0.382    | 
| BM1684 PCIe  | yolact_bmcv.pcie | yolact_bm1684_fp32_1b.bmodel  | 0.264         | 0.395    |
| BM1684 PCIe  | yolact_bmcv.pcie | yolact_bm1684_int8_1b.bmodel  | 0.254         | 0.384    |




> **测试说明**：  
> 1. batch_size=4和batch_size=1的模型精度一致；
> 2. SoC和PCIe的模型精度一致；
> 3. AP@IoU=0.5:0.95为area=all对应的指标。

## 7. 性能测试
### 7.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684/yolact_bm1684_fp32_1b.bmodel  
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

|                  测试模型                    | calculate time(ms)|
| ------------------------------------------- | ----------------- |
| BM1684X/yolact_bm1684x_fp32_1b.bmodel       | 212.68            |
| BM1684X/yolact_bm1684x_fp16_1b.bmodel       | 74.68             |
| BM1684X/yolact_bm1684x_int8_1b.bmodel       | 67.28             |
| BM1684X/yolact_bm1684x_int8_4b.bmodel       | 62.96             |
| BM1684/yolact_bm1684_fp32_1b.bmodel         | 114.60            |
| BM1684/yolact_bm1684_int8_1b.bmodel         | 135.59            | 
| BM1684/yolact_bm1684_int8_4b.bmodel         | 53.90             | 

> **测试说明**：  
> 1. 性能测试结果具有一定的波动性；
> 2. `calculate time`已折算为平均每张图片的推理时间；
> 3. SoC和PCIe的测试结果基本一致。


### 7.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++例程打印的预处理时间、推理时间、后处理时间为整个batch处理的时间，需除以相应的batch size才是每张图片的处理时间。

在不同的测试平台上，使用不同的例程、模型测试`datasets/val2017_1000`，conf_thresh=0.5，nms_thresh=0.5，性能测试结果如下：
|   测试平台   |     测试程序       |             测试模型                 |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------  | ---------------- | ----------------------------------- | --------  | ------------- | -----------  | ------------   |
| BM1684X SoC | yolact_bmcv.py   |    yolact_bm1684x_fp32_1b.bmodel    |   3.13    |    1.70       | 206.39       | 127.21         |
| BM1684X SoC | yolact_bmcv.py   |    yolact_bm1684x_fp16_1b.bmodel    |   3.07    |    1.70       | 64.73        | 127.66         |
| BM1684X SoC | yolact_bmcv.py   |    yolact_bm1684x_int8_1b.bmodel    |   3.08    |    1.70       | 56.25        | 124.91         |
| BM1684X SoC | yolact_bmcv.py   |    yolact_bm1684x_int8_4b.bmodel    |   2.93    |    1.60       | 52.75        | 124.43         |
| BM1684X SoC | yolact_opencv.py |    yolact_bm1684x_fp32_1b.bmodel    |   3.34    |    29.67      | 214.87       | 128.47         | 
| BM1684X SoC | yolact_opencv.py |    yolact_bm1684x_fp16_1b.bmodel    |   3.33    |    29.49      | 73.35        | 129.07         |
| BM1684X SoC | yolact_opencv.py |    yolact_bm1684x_int8_1b.bmodel    |   3.35    |    28.86      | 64.66        | 125.11         |
| BM1684X SoC | yolact_opencv.py |    yolact_bm1684x_int8_4b.bmodel    |   3.25    |    28.33      | 61.85        | 123.23         |
| BM1684X SoC | yolact_bmcv.soc  |    yolact_bm1684x_fp32_1b.bmodel    |   4.83    |    0.58       | 195.23       | 56.91          |  
| BM1684X SoC | yolact_bmcv.soc  |    yolact_bm1684x_fp16_1b.bmodel    |   4.81    |    0.58       | 53.60        | 56.91          |
| BM1684X SoC | yolact_bmcv.soc  |    yolact_bm1684x_int8_1b.bmodel    |   4.79    |    0.58       | 45.13        | 55.63          |
| BM1684X SoC | yolact_bmcv.soc  |    yolact_bm1684x_int8_4b.bmodel    |   4.73    |    0.55       | 43.88        | 62.58          | 
| BM1684 SoC  | yolact_bmcv.py   |    yolact_bm1684_fp32_1b.bmodel     |   3.67    |    2.29       | 108.78       | 138.67         | 
| BM1684 SoC  | yolact_bmcv.py   |    yolact_bm1684_int8_1b.bmodel     |   3.67    |    2.29       | 129.43       | 136.95         |  
| BM1684 SoC  | yolact_bmcv.py   |    yolact_bm1684_int8_4b.bmodel     |   3.45    |    2.16       | 45.12        | 124.43         |
| BM1684 SoC  | yolact_opencv.py |    yolact_bm1684_fp32_1b.bmodel     |   3.86    |    30.24      | 114.85       | 136.10         | 
| BM1684 SoC  | yolact_opencv.py |    yolact_bm1684_int8_1b.bmodel     |   3.85    |    29.33      | 135.56       | 135.47         | 
| BM1684 SoC  | yolact_opencv.py |    yolact_bm1684_int8_4b.bmodel     |   3.80    |    29.81      | 46.18        | 135.84         | 
| BM1684 SoC  | yolact_bmcv.soc  |    yolact_bm1684_fp32_1b.bmodel     |   5.36    |    1.51       | 97.97        | 56.93          | 
| BM1684 SoC  | yolact_bmcv.soc  |    yolact_bm1684_int8_1b.bmodel     |   5.43    |    1.51       | 118.52       | 56.35          | 
| BM1684 SoC  | yolact_bmcv.soc  |    yolact_bm1684_int8_4b.bmodel     |   5.30    |    1.46       | 38.025       | 63.15          | 

> **测试说明**：  
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. BM1684/1684X SoC的主控CPU均为8核 ARM A53 42320 DMIPS @2.3GHz，PCIe上的性能由于CPU的不同可能存在较大差异；
> 4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异，不同的阈值对后处理时间影响较大。 

## 8. FAQ
[Yolact移植相关问题可参考Yolact常见问题](./docs/Yolact_Common_Problems.md)，其他问题请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。