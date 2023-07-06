[简体中文](./README.md) | [English](./README_EN.md)

# YOLOv5_opt🚀

## 目录

* [1. 简介](#1-简介)
* [2. 特性](#2-特性)
* [3. 准备模型与数据](#3-准备模型与数据)
* [4. 模型编译](#4-模型编译)
  * [4.1 TPU-MLIR编译BModel](#41-tpu-mlir编译bmodel)
* [5. 例程测试](#5-例程测试)
* [6. 精度测试](#6-精度测试)
  * [6.1 测试方法](#61-测试方法)
  * [6.2 测试结果](#62-测试结果)
* [7. 性能测试](#7-性能测试)
  * [7.1 bmrt_test](#71-bmrt_test)
  * [7.2 程序运行性能](#72-程序运行性能)
* [8. FAQ](#8-faq)
  
## 1. 简介
本例程基于[YOLOv5](../YOLOv5/README.md)，在BM1684X上使用tpu_kernel的`tpu_kernel_api_yolov5_detect_out`算子对后处理进行加速，加速效果显著。

## 2. 特性
* 支持使用tpu_kernel进行后处理加速
* 支持BM1684X(x86 PCIe、SoC)
* 支持FP32、FP16(BM1684X)、INT8模型编译和推理
* 支持基于BMCV预处理C++推理
* 支持单batch和多batch模型推理
* 支持图片和视频测试
 
## 3. 准备模型与数据
建议使用TPU-MLIR编译BModel，Pytorch模型在编译前要导出成onnx模型。本例程暂不支持单输出模型，**建议采用多输出模型，性能更优。多输出是指源模型最后N个卷积层的输出(N <= 8)，导出方法参考[YOLOv5_tpukernel模型导出](./docs/YOLOv5_tpukernel_Export_Guide.md#)。**

​同时，您需要准备用于测试的数据集，如果量化模型，还要准备用于量化的数据集。

​本例程在`scripts`目录下提供了相关模型和数据集的下载脚本`download.sh`，您也可以自己准备模型和数据集，并参考[4. 模型编译](#4-模型编译)进行模型转换。

```bash
# 安装unzip，若已安装请跳过，非ubuntu系统视情况使用yum或其他方式安装
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```

下载的模型包括：
```
./models
├── BM1684X
│   ├── yolov5s_tpukernel_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=1
│   ├── yolov5s_tpukernel_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=1
│   ├── yolov5s_tpukernel_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=1
│   └── yolov5s_tpukernel_int8_4b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=4
└── onnx
    └── yolov5s_tpukernel.onnx             # 导出的onnx动态模型       
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

### 4.1 TPU-MLIR编译BModel
模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#2-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index/material/31/all.html)相应版本的SDK中获取)。

- 生成FP32 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（支持BM1684X），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684x
```

​执行上述命令会在`models/BM1684X/`下生成`yolov5s_tpukernel_fp32_1b.bmodel`文件，即转换好的FP32 BModel。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（支持BM1684X），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x
```

​执行上述命令会在`models/BM1684X/`下生成`yolov5s_tpukernel_fp16_1b.bmodel`文件，即转换好的FP16 BModel。

- 生成INT8 BModel

​本例程在`scripts`目录下提供了量化INT8 BModel的脚本，请注意修改`gen_int8bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台（支持BM1684X），如：

```shell
./scripts/gen_int8bmodel_mlir.sh bm1684x
```

​上述脚本会在`models/BM1684X`下生成`yolov5s_tpukernel_int8_1b.bmodel`等文件，即转换好的INT8 BModel。


## 5. 例程测试
- [C++例程](./cpp/README.md)
- [Python例程](./python/README.md)
## 6. 精度测试
### 6.1 测试方法

首先，参考[C++例程](cpp/README.md#32-测试图片)或者[Python例程](python/README.md#22-测试图片)推理要测试的数据集，生成预测的json文件，注意修改数据集(datasets/coco/val2017_1000)和相关参数(conf_thresh=0.1、nms_thresh=0.6)。  
然后，使用`tools`目录下的`eval_coco.py`脚本，将测试生成的json文件与测试集标签json文件进行对比，计算出目标检测的评价指标，命令如下：
```bash
# 安装pycocotools，若已安装请跳过
pip3 install pycocotools
# 请根据实际情况修改程序路径和json文件路径
python3 tools/eval_coco.py --gt_path datasets/coco/instances_val2017_1000.json --result_json cpp/yolov5_bmcv/results/yolov5s_tpukernel_fp32_1b.bmodel_val2017_1000_bmcv_cpp_result.json
```
### 6.2 测试结果
在coco/val2017_1000数据集上，精度测试结果如下：
|   测试平台    |      测试程序     |              测试模型               |AP@IoU=0.5:0.95|AP@IoU=0.5|
| ------------ | ---------------- | ---------------------------------   | ------------- | -------- |
| BM1684X PCIe | yolov5_opencv.py | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.353         | 0.536    |
| BM1684X PCIe | yolov5_opencv.py | yolov5s_v6.1_3output_fp16_1b.bmodel | 0.353         | 0.536    |
| BM1684X PCIe | yolov5_opencv.py | yolov5s_v6.1_3output_int8_1b.bmodel | 0.339         | 0.527    |
| BM1684X PCIe | yolov5_bmcv.py   | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.351         | 0.532    |
| BM1684X PCIe | yolov5_bmcv.py   | yolov5s_v6.1_3output_fp16_1b.bmodel | 0.351         | 0.532   |
| BM1684X PCIe | yolov5_bmcv.py   | yolov5s_v6.1_3output_int8_1b.bmodel | 0.334         | 0.520    |
| BM1684X PCIe | yolov5_bmcv.pcie | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.351         | 0.536    |
| BM1684X PCIe | yolov5_bmcv.pcie | yolov5s_v6.1_3output_fp16_1b.bmodel | 0.351         | 0.535    |
| BM1684X PCIe | yolov5_bmcv.pcie | yolov5s_v6.1_3output_int8_1b.bmodel | 0.337         | 0.526    |
| BM1684X PCIe | yolov5_sail.pcie | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.351         | 0.536    |
| BM1684X PCIe | yolov5_sail.pcie | yolov5s_v6.1_3output_fp16_1b.bmodel | 0.351         | 0.535    |
| BM1684X PCIe | yolov5_sail.pcie | yolov5s_v6.1_3output_int8_1b.bmodel | 0.337         | 0.526    |

> **测试说明**：  
> 1. batch_size=4和batch_size=1的模型精度一致；
> 2. SoC和PCIe的模型精度一致；
> 3. AP@IoU=0.5:0.95为area=all对应的指标。
> 4. 为避免因检测框输出太多导致TPU内存超限，这里强制设置`conf_thresh和nms_thresh>=0.1`，同样的参数下`(nms_thresh=0.6，conf_thresh=0.1)`，[原YOLOv5例程](../YOLOv5/README.md)fp32模型的`AP@IoU=0.5:0.95=0.345`。
## 7. 性能测试
### 7.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684X/yolov5s_tpukernel_fp32_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

|                  测试模型                   | calculate time(ms) |
| ------------------------------------------- | ----------------- |
| BM1684X/yolov5s_tpukernel_fp32_1b.bmodel | 19.6              |
| BM1684X/yolov5s_tpukernel_fp16_1b.bmodel | 6.2               |
| BM1684X/yolov5s_tpukernel_int8_1b.bmodel | 3.4               |
| BM1684X/yolov5s_tpukernel_int8_4b.bmodel | 3.2               |

> **测试说明**：  
> 1. 性能测试结果具有一定的波动性；
> 2. `calculate time`已折算为平均每张图片的推理时间；
> 3. SoC和PCIe的测试结果基本一致。

### 7.2 程序运行性能
参考[C++例程](cpp/README.md#3-推理测试)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++例程打印的预处理时间、推理时间、后处理时间为整个batch处理的时间，需除以相应的batch size才是每张图片的处理时间。

在不同的测试平台上，使用不同的例程、模型测试`datasets/coco/val2017_1000`，conf_thresh=0.5，nms_thresh=0.5，性能测试结果如下：
|    测试平台  |     测试程序      |             测试模型            |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ---------------- | --------------------------------| -------- | -------------- | ---------      | --------- |
| BM1684X SoC | yolov5_opencv.py | yolov5s_tpukernel_fp32_1b.bmodel | 15.0     | 22.4          | 36.16          | 2.18      |
| BM1684X SoC | yolov5_opencv.py | yolov5s_tpukernel_fp16_1b.bmodel | 15.0     | 22.4          | 22.74          | 2.18      |
| BM1684X SoC | yolov5_opencv.py | yolov5s_tpukernel_int8_1b.bmodel | 15.0     | 22.4          | 20.16          | 2.18      |
| BM1684X SoC | yolov5_opencv.py | yolov5s_tpukernel_int8_4b.bmodel | 15.0     | 23.1          | 5.03           | 2.18      |
| BM1684X SoC | yolov5_bmcv.py   | yolov5s_tpukernel_fp32_1b.bmodel | 3.1      | 2.4           | 25.42          | 2.18      |
| BM1684X SoC | yolov5_bmcv.py   | yolov5s_tpukernel_fp16_1b.bmodel | 3.1      | 2.4           | 11.92          | 2.18      |
| BM1684X SoC | yolov5_bmcv.py   | yolov5s_tpukernel_int8_1b.bmodel | 3.1      | 2.4           | 9.05           | 2.18      |
| BM1684X SoC | yolov5_bmcv.py   | yolov5s_tpukernel_int8_4b.bmodel | 2.9      | 2.3           | 8.21           | 2.18      |
| BM1684X SoC | yolov5_bmcv.soc  | yolov5s_tpukernel_fp32_1b.bmodel | 4.7      | 0.8           | 19.67          | 1.20      |
| BM1684X SoC | yolov5_bmcv.soc  | yolov5s_tpukernel_fp16_1b.bmodel | 4.7      | 0.8           | 6.27           | 1.20      |
| BM1684X SoC | yolov5_bmcv.soc  | yolov5s_tpukernel_int8_1b.bmodel | 4.7      | 0.8           | 3.40           | 1.20      |
| BM1684X SoC | yolov5_bmcv.soc  | yolov5s_tpukernel_int8_4b.bmodel | 4.7      | 0.8           | 3.17           | 1.20      |
| BM1684X SoC | yolov5_sail.soc  | yolov5s_tpukernel_fp32_1b.bmodel | 2.8      | 3.1           | 19.70          | 1.38      |
| BM1684X SoC | yolov5_sail.soc  | yolov5s_tpukernel_fp16_1b.bmodel | 2.8      | 3.1           | 6.30           | 1.38      |
| BM1684X SoC | yolov5_sail.soc  | yolov5s_tpukernel_int8_1b.bmodel | 2.8      | 3.1           | 3.41           | 1.38      |
| BM1684X SoC | yolov5_sail.soc  | yolov5s_tpukernel_int8_4b.bmodel | 2.6      | 2.5           | 3.18           | 1.38      |

> **测试说明**：  
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. 1684X SoC的主控CPU均为8核 ARM A53 42320 DMIPS @2.3GHz，PCIe上的性能由于CPU的不同可能存在较大差异；
> 4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异，不同的阈值对后处理时间影响较大。 

## 8. FAQ
请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。