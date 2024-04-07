[简体中文](./README.md) | [English](./README_EN.md)

# ppYoloe

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
ppyoloe是百度提出的基于PP-YOLOv2的卓越的单阶段Anchor-free模型，超越了多种流行的YOLO模型。

**论文地址** (https://arxiv.org/pdf/2203.16250.pdf)

**官方源码地址** (https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/ppyoloe)

## 2. 特性
* 支持BM1684X(x86 PCIe、SoC)和BM1684(x86 PCIe、SoC、arm PCIe)
* 支持FP32、FP16(BM1684X)模型编译和推理
* 支持基于BMCV、sail预处理的C++推理
* 支持基于OpenCV和BMCV预处理的Python推理
* 支持单batch和多batch模型推理
* 支持图片和视频测试

## 3. 准备模型与数据
推荐您使用新版编译工具链TPU-MLIR编译BModel，目前直接支持的框架有ONNX、Caffe和TFLite，其他框架的模型需要转换成onnx模型。如何将其他深度学习架构的网络模型转换成onnx, 可以参考onnx官网: https://github.com/onnx/tutorials ；ppyoloe模型导出为onnx的方法可参考PaddleDetection官方说明：https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.6/deploy/EXPORT_ONNX_MODEL.md 和 https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.6/deploy/end2end_ppyoloe/README.md

本demo使用的模型为官方的[ppyoloe_crn_s_400e_coco](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_s_400e_coco.pdparams)，由于模型的后处理较复杂，使用TPU-MLIR将该模型编译为BModel时，指定了模型的输出为模型中间有关目标框的两个输出，p2o.Div.1, p2o.Concat.29，后处理部分则在部署程序中实现。若在转换Bmodel模型要添加参考输入以判断转换的正确性，可以参考[多输入模型npz验证文件制作](./docs/prepare_npz.md)文档。

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
│   └── ppyoloe_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684的FP32 BModel，batch_size=1
├── BM1684X
│   ├── ppyoloe_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=1
│   └── ppyoloe_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=1
└── onnx
    └── ppyoloe.onnx             # 导出的onnx动态模型
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

本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684/BM1684X**），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684
#or
./scripts/gen_fp32bmodel_mlir.sh bm1684x
```

执行上述命令会在`models/BM1684`或`models/BM1684X/`下生成`ppyoloe_fp32_1b.bmodel`文件，即转换好的FP32 BModel。

- 生成FP16 BModel

本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X**），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x
```

执行上述命令会在`models/BM1684X/`下生成`ppyoloe_fp16_1b.bmodel`文件，即转换好的FP16 BModel。


## 5. 例程测试
- [C++例程](./cpp/README.md)
- [Python例程](./python/README.md)

## 6. 精度测试
### 6.1 测试方法

首先，参考[C++例程](cpp/README.md#32-测试图片)或[Python例程](python/README.md#22-测试图片)推理要测试的数据集，生成预测的json文件，注意修改数据集(datasets/coco/val2017_1000)和相关参数(conf_thresh=0.4、nms_thresh=0.6)。  
然后，使用`tools`目录下的`eval_coco.py`脚本，将测试生成的json文件与测试集标签json文件进行对比，计算出目标检测的评价指标，命令如下：

```bash
# 安装pycocotools，若已安装请跳过
pip3 install pycocotools
# 请根据实际情况修改程序路径和json文件路径
python3 tools/eval_coco.py --gt_path datasets/coco/instances_val2017_1000.json --result_json results/ppyoloe_fp32_1b.bmodel_val2017_1000_bmcv_python_result.json
```

### 6.2 测试结果

在coco2017val_1000数据集上，精度测试结果如下：

|   测试平台    |      测试程序     |        测试模型        | AP@IoU=0.5:0.95 | AP@IoU=0.5 |
| ------------ | ---------------- | ---------------------- | --------------- | ---------- |
| BM1684 PCIe  | ppyoloe_opencv.py | ppyoloe_fp32_1b.bmodel | 0.377 	     | 0.508      |
| BM1684 PCIe  | ppyoloe_bmcv.py   | ppyoloe_fp32_1b.bmodel | 0.380 	     | 0.513      |
| BM1684 PCIe  | ppyoloe_bmcv.pcie | ppyoloe_fp32_1b.bmodel | 0.378        | 0.510      |
| BM1684 PCIe  | ppyoloe_sail.pcie | ppyoloe_fp32_1b.bmodel | 0.378        | 0.510      |
| BM1684X PCIe | ppyoloe_opencv.py | ppyoloe_fp32_1b.bmodel | 0.377 	     | 0.508 	    |
| BM1684X PCIe | ppyoloe_opencv.py | ppyoloe_fp16_1b.bmodel | 0.377 	     | 0.508 	    |
| BM1684X PCIe | ppyoloe_bmcv.py   | ppyoloe_fp32_1b.bmodel | 0.380 	     | 0.513  	  |
| BM1684X PCIe | ppyoloe_bmcv.py   | ppyoloe_fp16_1b.bmodel | 0.380 	     | 0.513 	    |
| BM1684X PCIe | ppyoloe_bmcv.pcie | ppyoloe_fp32_1b.bmodel | 0.379 	     | 0.510 	    |
| BM1684X PCIe | ppyoloe_bmcv.pcie | ppyoloe_fp16_1b.bmodel | 0.378 	     | 0.510 	    |
| BM1684X PCIe | ppyoloe_sail.pcie | ppyoloe_fp32_1b.bmodel | 0.379 	     | 0.510 	    |
| BM1684X PCIe | ppyoloe_sail.pcie | ppyoloe_fp16_1b.bmodel | 0.378 	     | 0.510 	    |

> **测试说明**：  
>
> 1. SoC和PCIe的模型精度一致；
> 2. AP@IoU=0.5:0.95为area=all对应的指标。

## 7. 性能测试
### 7.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684/ppyoloe_fp32_1b.bmodel
```
测试结果中的`calculate time()`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

|                  测试模型                   | calculate time(ms) |
| ------------------------------------------- | ----------------- |
| BM1684/ppyoloe_fp32_1b.bmodel  	            |       26.01       |
| BM1684/ppyoloe_fp32_4b.bmodel  	            |       25.62       |
| BM1684X/ppyoloe_fp32_1b.bmodel 	            |       35.80       |
| BM1684X/ppyoloe_fp32_4b.bmodel 	            |       35.15       |
| BM1684X/ppyoloe_fp16_1b.bmodel 	            |       10.12       |
| BM1684X/ppyoloe_fp16_4b.bmodel 	            |       8.90        |

> **测试说明**：  
>
> 1. 性能测试结果具有一定的波动性；
> 2. `calculate time`已折算为平均每张图片的推理时间；
> 3. SoC和PCIe的测试结果基本一致。

### 7.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++例程打印的预处理时间、推理时间、后处理时间为整个batch处理的时间，需除以相应的batch size才是每张图片的处理时间。

在不同的测试平台上，使用不同的例程、模型测试`datasets/val2017_1000`，conf_thresh=0.4，nms_thresh=0.6，性能测试结果如下：
|    测试平台  |     测试程序      |     测试模型          |decode_time|preprocess_time|inference_time|postprocess_time|
| ----------- | ---------------- | ---------------------- | -------- | -------------- | ---------     | --------- |
| BM1684 SoC  | yolox_opencv.py  | ppyoloe_fp32_1b.bmodel | 15.19    | 45.22          | 45.78         | 12.86     |
| BM1684 SoC  | yolox_bmcv.py    | ppyoloe_fp32_1b.bmodel | 6.82     | 3.58           | 33.72         | 12.83     |
| BM1684 SoC  | yolox_bmcv.soc   | ppyoloe_fp32_1b.bmodel | 5.02     | 1.72           | 30.78         | 16.91     |
| BM1684 SoC  | ppyoloeail.soc   | ppyoloe_fp32_1b.bmodel | 3.22     | 4.19           | 31.18         | 7.99      |
| BM1684X SoC | yolox_opencv.py  | ppyoloe_fp32_1b.bmodel | 3.37     | 41.60          | 43.79         | 12.75     |
| BM1684X SoC | yolox_opencv.py  | ppyoloe_fp16_1b.bmodel | 3.21     | 40.63          | 24.48         | 12.62     |
| BM1684X SoC | yolox_bmcv.py    | ppyoloe_fp32_1b.bmodel | 3.05     | 2.68           | 30.28         | 13.09     |
| BM1684X SoC | yolox_bmcv.py    | ppyoloe_fp16_1b.bmodel | 3.07     | 2.68           | 11.09         | 13.10     |
| BM1684X SoC | yolox_bmcv.soc   | ppyoloe_fp32_1b.bmodel | 4.42     | 0.98           | 27.03         | 8.64      |
| BM1684X SoC | yolox_bmcv.soc   | ppyoloe_fp16_1b.bmodel | 4.48     | 1.00           | 7.88          | 8.69      |
| BM1684X SoC | ppyoloeail.soc   | ppyoloe_fp32_1b.bmodel | 2.78     | 3.28           | 27.44         | 8.04      |
| BM1684X SoC | ppyoloeail.soc   | ppyoloe_fp16_1b.bmodel | 2.69     | 3.28           | 8.20          | 8.06      |

> **测试说明**：  
>
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. BM1684 SoC的测试平台为标准版SE5，BM1684X SoC的测试平台为标准版SE7
> 4. BM1684/1684X SoC的主控CPU均为8核 ARM A53 42320 DMIPS @2.3GHz，PCIe上的性能由于CPU的不同可能存在较大差异；
> 5. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异，不同的阈值对后处理时间影响较大。 

## 8. FAQ
[常见问题解答](../../docs/FAQ.md)
