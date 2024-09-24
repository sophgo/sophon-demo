[简体中文](./README.md) | [English](./README_EN.md)

# YOLOv5

## 目录

- [YOLOv5](#yolov5)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
    - [2.1 目录结构说明](#21-目录结构说明)
    - [2.2 SDK特性](#22-sdk特性)
  - [3. 数据准备与模型编译](#3-数据准备与模型编译)
    - [3.1 数据准备](#31-数据准备)
  - [4. 例程测试](#4-例程测试)
  - [5. 精度测试](#5-精度测试)
    - [5.1 测试方法](#51-测试方法)
    - [5.2 测试结果](#52-测试结果)
  - [6. 性能测试](#6-性能测试)
    - [6.1 tpurt_test](#61-tpurt_test)
    - [6.2 程序运行性能](#62-程序运行性能)
  - [7. FAQ](#7-faq)
  
## 1. 简介
​YOLOv5是非常经典的基于anchor的One Stage目标检测算法，因其优秀的精度和速度表现，在工程实践应用中获得了非常广泛的应用。本例程对[​YOLOv5官方开源仓库](https://github.com/ultralytics/yolov5)v6.1版本的模型和算法进行移植，使之能在SOPHON BM1690上进行推理测试。

## 2. 特性

### 2.1 目录结构说明
```bash
├── docs                  # 存放本例程专用文档，如ONNX导出、移植常见问题等
├── pics                  # 存放README等说明文档中用到的图片
├── python                # 存放Python例程及其README
|   ├──README_EN.md 
|   ├──README.md 
|   ├──yolov5_opencv.py   # 使用OpenCV解码、OpenCV前处理、SAIL推理的Python例程
|   └──...                # Python例程共用功能的封装。
├── README_EN.md          # 本例程的英文指南
├── README.md             # 本例程的中文指南
├── scripts               # 存放模型编译、数据下载、自动测试等shell脚本
└── tools                 # 存放精度测试、性能比对等python脚本
```

### 2.2 SDK特性
* 支持BM1690(PCIe)
* 支持FP32、FP16、INT8模型推理
* 支持基于OpenCV的Python推理
* 支持单batch和多batch模型推理
* 支持1个输出和3个输出模型推理
* 支持图片和视频测试

> **注意：**  
> 本例程支持三输出以及单输出模型，其中单输出模型性能更高，但是量化需要设置敏感层；三输出模型量化简单，在**用于验证模型准确性时，推荐使用三输出模型**

## 3. 数据准备与模型编译

### 3.1 数据准备

​本例程在`scripts`目录下提供了相关模型和数据的下载脚本`download.sh`，**如果您希望自己准备模型和数据集，可以跳过本小节，参考[3.2 模型编译](#32-模型编译)进行模型转换。**

```bash
chmod -R +x scripts/
./scripts/download.sh
```

下载的模型包括：
```
./models
└── BM1690
    ├── yolov5s_v6.1_3output_fp16_1b.bmodel       # 使用TPU-MLIR编译，用于BM1690的FP16 BModel，batch_size=1, num_core=1
    ├── yolov5s_v6.1_3output_fp32_1b.bmodel       # 使用TPU-MLIR编译，用于BM1690的FP32 BModel，batch_size=1, num_core=1
    ├── yolov5s_v6.1_3output_int8_1b.bmodel       # 使用TPU-MLIR编译，用于BM1690的INT8 BModel，batch_size=1, num_core=1
    ├── yolov5s_v6.1_3output_int8_4b.bmodel       # 使用TPU-MLIR编译，用于BM1690的INT8 BModel，batch_size=1, num_core=1
    ├── yolov5s_v6.1_3output_fp16_1b_8core.bmodel # 使用TPU-MLIR编译，用于BM1690的FP16 BModel，batch_size=1, num_core=8
    ├── yolov5s_v6.1_3output_fp32_1b_8core.bmodel # 使用TPU-MLIR编译，用于BM1690的FP32 BModel，batch_size=1, num_core=8
    ├── yolov5s_v6.1_3output_int8_1b_8core.bmodel # 使用TPU-MLIR编译，用于BM1690的INT8 BModel，batch_size=1, num_core=8
    └── yolov5s_v6.1_3output_int8_4b_8core.bmodel # 使用TPU-MLIR编译，用于BM1690的INT8 BModel，batch_size=4, num_core=8
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

## 4. 例程测试
- [Python例程](./python/README.md)

## 5. 精度测试
### 5.1 测试方法

首先，参考[C++例程](cpp/README.md#32-测试图片)或[Python例程](python/README.md#22-测试图片)推理要测试的数据集，生成预测的json文件，注意修改数据集(datasets/coco/val2017_1000)和相关参数(conf_thresh=0.001、nms_thresh=0.6)。  
然后，使用`tools`目录下的`eval_coco.py`脚本，将测试生成的json文件与测试集标签json文件进行对比，计算出目标检测的评价指标，命令如下：
```bash
# 安装pycocotools，若已安装请跳过
pip3 install pycocotools
# 请根据实际情况修改程序路径和json文件路径
python3 tools/eval_coco.py --gt_path datasets/coco/instances_val2017_1000.json --result_json results/yolov5s_v6.1_3output_fp32_1b.bmodel_val2017_1000_opencv_python_result.json
```
### 5.2 测试结果
CPP设置`--use_cpu_opt=false`或python不设置`--use_cpu_opt`进行测试，在`datasets/coco/val2017_1000`数据集上，精度测试结果如下：
|   测试平台    |      测试程序     |              测试模型               |AP@IoU=0.5:0.95|AP@IoU=0.5|
| ------------ | ---------------- | ----------------------------------- | ------------- | -------- |
|     BM1690 PCIe     | yolov5_opencv.py   | yolov5s_v6.1_3output_fp32_1b.bmodel      |    0.377 |    0.580 |
|     BM1690 PCIe     | yolov5_opencv.py   | yolov5s_v6.1_3output_int8_1b.bmodel      |    0.355 |    0.567 |
|     BM1690 PCIe     | yolov5_opencv.py   | yolov5s_v6.1_3output_int8_4b.bmodel      |    0.355 |    0.571 |

> **测试说明**：  
> 1. 由于sdk版本之间可能存在差异，实际运行结果与本表有<0.01的精度误差是正常的；
> 2. AP@IoU=0.5:0.95为area=all对应的指标；

## 6. 性能测试
### 6.1 tpurt_test
使用tpu-model-rt测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
tpu-model-rt --bmodel models/BM1690/yolov5s_v6.1_3output_fp32_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

|                  测试模型                         | calculate time(ms) |
| -------------------------------------------       | ----------------- |
| BM1690/yolov5s_v6.1_3output_fp32_1b.bmodel |          17.88  |
| BM1690/yolov5s_v6.1_3output_int8_1b.bmodel |          2.56  |
| BM1690/yolov5s_v6.1_3output_int8_4b.bmodel |           2.06  |

> **测试说明**：  
> 1. 性能测试结果具有一定的波动性；
> 2. `calculate time`已折算为平均每张图片的推理时间；

### 6.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++和Python例程打印的时间已经折算为单张图片的处理时间。

CPP设置`--use_cpu_opt=false`或python不设置`--use_cpu_opt`进行测试，在不同的测试平台上，使用不同的例程、模型测试`datasets/coco/val2017_1000`，conf_thresh=0.5，nms_thresh=0.5，性能测试结果如下：
|    测试平台  |     测试程序      |             测试模型                |decode_time    |preprocess_time  |inference_time   |postprocess_time| 
| ----------- | ---------------- | ----------------------------------- | --------      | ---------       | ---------        | --------- |
|   BM1690 PCIe   | yolov5_opencv.py  |yolov5s_v6.1_3output_fp32_1b.bmodel|      4.06      |      10.63      |      31.40      |     15.71      |
|   BM1690 PCIe   | yolov5_opencv.py  |yolov5s_v6.1_3output_int8_1b.bmodel|      2.87      |      7.97      |      33.02      |     12.89      |
|   BM1690 PCIe   | yolov5_opencv.py  |yolov5s_v6.1_3output_int8_4b.bmodel|      2.48      |      5.13      |      27.53      |     9.66      |

> **测试说明**：  
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. PCIe上的性能由于处理器的不同可能存在较大差异；
> 4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异，不同的阈值对后处理时间影响较大。 

## 7. FAQ
YOLOv5移植相关问题可参考[YOLOv5常见问题](./docs/YOLOv5_Common_Problems.md)，其他问题请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。