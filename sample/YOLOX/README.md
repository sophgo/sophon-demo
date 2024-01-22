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
* [9. 致谢](#9-致谢)
  
## 1. 简介
YOLOx由旷世研究提出,是基于YOLO系列的改进，引入了解耦头和Anchor-free，提高算法整体的检测性能

**论文地址** (https://arxiv.org/abs/2107.08430)

**官方源码地址** (https://github.com/Megvii-BaseDetection/YOLOX)

## 2. 特性
* 支持BM1688(SoC)、支持BM1684X(x86 PCIe、SoC)和BM1684(x86 PCIe、SoC、arm PCIe)
* 支持FP32、FP16(BM1684X/BM1688)、INT8模型编译和推理
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
├── BM1688
│   ├── yolox_s_fp32_1b.bmodel         # 使用TPU-MLIR编译，用于BM1688的单核FP32 BModel，batch_size=1
│   ├── yolox_s_fp32_1b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的双核FP32 BModel，batch_size=1
│   ├── yolox_s_fp16_1b.bmodel         # 使用TPU-MLIR编译，用于BM1688的单核FP16 BModel，batch_size=1
│   ├── yolox_s_fp16_1b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的双核FP16 BModel，batch_size=1
│   ├── yolox_s_int8_1b.bmodel         # 使用TPU-MLIR编译，用于BM1688的单核INT8 BModel，batch_size=1
│   ├── yolox_s_int8_4b.bmodel         # 使用TPU-MLIR编译，用于BM1688的单核INT8 BModel，batch_size=4
│   ├── yolox_s_int8_1b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的单核INT8 BModel，batch_size=1
│   └── yolox_s_int8_4b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的双核INT8 BModel，batch_size=4
│── torch
│   ├── yolox_s.pt               # 源模型
|   └── yolox_s.torchscript.pt   # 源模型trace后的torchscript模型
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

​本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684/BM1684X/BM1688**），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684 #bm1684x/bm1688
```

​执行上述命令会在`models/BM1684`等文件夹下生成`yolox_s_fp32_1b.bmodel`文件，即转换好的FP32 BModel。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688**），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x #bm1688
```

​执行上述命令会在`models/BM1684X/`等文件夹下生成`yolox_s_fp16_1b.bmodel`文件，即转换好的FP16 BModel。

- 生成INT8 BModel

​本例程在`scripts`目录下提供了量化INT8 BModel的脚本，请注意修改`gen_int8bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台（**支持BM1684/BM1684X/BM1688**），如：

```shell
./scripts/gen_int8bmodel_mlir.sh bm1684 #bm1684x/bm1688
```

​上述脚本会在`models/BM1684`等文件夹下生成`yolox_s_int8_1b.bmodel`等文件，即转换好的INT8 BModel。


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
在`datasets/coco/val2017_1000`数据集上，参数设置为`nms_thresh=0.6,conf_thresh=0.001`,精度测试结果如下：
|   测试平台    |      测试程序     |        测试模型        | AP@IoU=0.5:0.95 | AP@IoU=0.5 |
| ------------ | ---------------- | ---------------------- | --------------- | ---------- |
| BM1684 PCIe  | yolox_opencv.py  | yolox_s_fp32_1b.bmodel |      0.403      |   0.590    |
| BM1684 PCIe  | yolox_opencv.py  | yolox_s_int8_1b.bmodel |      0.397      |   0.583    |
| BM1684 PCIe  | yolox_bmcv.py    | yolox_s_fp32_1b.bmodel |      0.402      |   0.590    |
| BM1684 PCIe  | yolox_bmcv.py    | yolox_s_int8_1b.bmodel |      0.397      |   0.582    |
| BM1684 PCIe  | yolox_bmcv.pcie  | yolox_s_fp32_1b.bmodel |      0.400      |   0.594    |
| BM1684 PCIe  | yolox_bmcv.pcie  | yolox_s_int8_1b.bmodel |      0.396      |   0.587    |
| BM1684 PCIe  | yolox_sail.pcie  | yolox_s_fp32_1b.bmodel |      0.400      |   0.594    |
| BM1684 PCIe  | yolox_sail.pcie  | yolox_s_int8_1b.bmodel |      0.396      |   0.587    |
| BM1684X PCIe | yolox_opencv.py  | yolox_s_fp32_1b.bmodel |      0.402      |   0.590    |
| BM1684X PCIe | yolox_opencv.py  | yolox_s_fp16_1b.bmodel |      0.402      |   0.590    |
| BM1684X PCIe | yolox_opencv.py  | yolox_s_int8_1b.bmodel |      0.402      |   0.587    |
| BM1684X PCIe | yolox_bmcv.py    | yolox_s_fp32_1b.bmodel |      0.402      |   0.590    |
| BM1684X PCIe | yolox_bmcv.py    | yolox_s_fp16_1b.bmodel |      0.402      |   0.590    |
| BM1684X PCIe | yolox_bmcv.py    | yolox_s_int8_1b.bmodel |      0.402      |   0.586    |
| BM1684X PCIe | yolox_bmcv.pcie  | yolox_s_fp32_1b.bmodel |      0.400      |   0.594    |
| BM1684X PCIe | yolox_bmcv.pcie  | yolox_s_fp16_1b.bmodel |      0.400      |   0.594    |
| BM1684X PCIe | yolox_bmcv.pcie  | yolox_s_int8_1b.bmodel |      0.401      |   0.592    |
| BM1684X PCIe | yolox_sail.pcie  | yolox_s_fp32_1b.bmodel |      0.400      |   0.594    |
| BM1684X PCIe | yolox_sail.pcie  | yolox_s_fp16_1b.bmodel |      0.400      |   0.594    |
| BM1684X PCIe | yolox_sail.pcie  | yolox_s_int8_1b.bmodel |      0.401      |   0.592    |
| BM1688 SoC   | yolox_opencv.py  | yolox_s_fp32_1b.bmodel |      0.403      |   0.590    |
| BM1688 SoC   | yolox_opencv.py  | yolox_s_fp16_1b.bmodel |      0.402      |   0.590    |
| BM1688 SoC   | yolox_opencv.py  | yolox_s_int8_1b.bmodel |      0.402      |   0.587    |
| BM1688 SoC   | yolox_bmcv.py    | yolox_s_fp32_1b.bmodel |      0.402      |   0.590    |
| BM1688 SoC   | yolox_bmcv.py    | yolox_s_fp16_1b.bmodel |      0.402      |   0.590    |
| BM1688 SoC   | yolox_bmcv.py    | yolox_s_int8_1b.bmodel |      0.402      |   0.587    |
| BM1688 SoC   | yolox_bmcv.soc   | yolox_s_fp32_1b.bmodel |      0.400      |   0.594    |
| BM1688 SoC   | yolox_bmcv.soc   | yolox_s_fp16_1b.bmodel |      0.400      |   0.594    |
| BM1688 SoC   | yolox_bmcv.soc   | yolox_s_int8_1b.bmodel |      0.402      |   0.592    |
| BM1688 SoC   | yolox_sail.soc   | yolox_s_fp32_1b.bmodel |      0.400      |   0.594    |
| BM1688 SoC   | yolox_sail.soc   | yolox_s_fp16_1b.bmodel |      0.400      |   0.594    |
| BM1688 SoC   | yolox_sail.soc   | yolox_s_int8_1b.bmodel |      0.402      |   0.592    |

> **测试说明**：  
> 1. 相同的程序、模型在SoC和PCIe上精度一致，batch_size=4和batch_size=1的模型精度一致，BM1688双核模型精度测试与单核模型精度一致；
> 2. AP@IoU=0.5:0.95为area=all对应的指标；
> 3. 由于sdk版本之间可能存在差异，实际运行结果与本表有<0.01的精度误差是正常的；

## 7. 性能测试
### 7.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684/yolox_s_fp32_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

|          测试模型                   | calculate time(ms) |
| ------------------------------      | ----------------- |
| BM1684/yolox_s_fp32_1b.bmodel       |     26.01         |
| BM1684/yolox_s_fp32_4b.bmodel       |     25.62         |
| BM1684/yolox_s_int8_1b.bmodel       |     19.54         |
| BM1684/yolox_s_int8_4b.bmodel       |     8.975         |
| BM1684X/yolox_s_fp32_1b.bmodel      |     27.92         |
| BM1684X/yolox_s_fp32_4b.bmodel      |     25.63         |
| BM1684X/yolox_s_fp16_1b.bmodel      |     6.27          |
| BM1684X/yolox_s_fp16_4b.bmodel      |     6.15          |
| BM1684X/yolox_s_int8_1b.bmodel      |     4.55          |
| BM1684X/yolox_s_int8_4b.bmodel      |     4.28          |
| BM1688/yolox_s_fp32_1b.bmodel       |     155.60        |
| BM1688/yolox_s_fp16_1b.bmodel       |     36.11         |
| BM1688/yolox_s_int8_1b.bmodel       |     21.44         |
| BM1688/yolox_s_int8_4b.bmodel       |     20.40         |
| BM1688/yolox_s_fp32_1b_2core.bmodel |     104.13        |
| BM1688/yolox_s_fp16_1b_2core.bmodel |     23.58         |
| BM1688/yolox_s_int8_1b_2core.bmodel |     15.97         |
| BM1688/yolox_s_int8_4b_2core.bmodel |     11.92         |

> **测试说明**：  
> 1. 性能测试结果具有一定的波动性；
> 2. `calculate time`已折算为平均每张图片的推理时间；
> 3. SoC和PCIe的测试结果基本一致；

### 7.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++例程打印的预处理时间、推理时间、后处理时间为整个batch处理的时间，需除以相应的batch size才是每张图片的处理时间。

在不同的测试平台上，使用不同的例程、模型测试`datasets/coco/val2017_1000`，`conf_thresh=0.5, nms_thresh=0.5`，性能测试结果如下：
|    测试平台  |     测试程序      |     测试模型          |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ---------------- | --------------------- | --------  | -------------- | ---------     | --------- |
| BM1684 SoC  | yolox_opencv.py  | yolox_s_fp32_1b.bmodel | 15.18    | 3.63           | 39.70         | 2.71     |
| BM1684 SoC  | yolox_opencv.py  | yolox_s_int8_1b.bmodel | 15.20    | 3.64           | 33.27         | 2.69     |
| BM1684 SoC  | yolox_opencv.py  | yolox_s_int8_4b.bmodel | 15.20    | 5.54           | 22.66         | 2.36     |
| BM1684 SoC  | yolox_bmcv.py    | yolox_s_fp32_1b.bmodel | 3.70     | 2.88           | 28.16         | 2.72     |
| BM1684 SoC  | yolox_bmcv.py    | yolox_s_int8_1b.bmodel | 3.50     | 2.22           | 21.62         | 2.71     |
| BM1684 SoC  | yolox_bmcv.py    | yolox_s_int8_4b.bmodel | 3.38     | 2.06           | 10.68         | 2.35     |
| BM1684 SoC  | yolox_bmcv.soc   | yolox_s_fp32_1b.bmodel | 4.86     | 1.47           | 25.78         | 2.68     |
| BM1684 SoC  | yolox_bmcv.soc   | yolox_s_int8_1b.bmodel | 4.57     | 1.46           | 19.44         | 2.68     |
| BM1684 SoC  | yolox_bmcv.soc   | yolox_s_int8_4b.bmodel | 4.76     | 1.41           | 8.96          | 2.07     |
| BM1684 SoC  | yolox_sail.soc   | yolox_s_fp32_1b.bmodel | 3.22     | 3.11           | 26.16         | 2.08     |
| BM1684 SoC  | yolox_sail.soc   | yolox_s_int8_1b.bmodel | 3.21     | 3.11           | 19.83         | 2.07     |
| BM1684 SoC  | yolox_sail.soc   | yolox_s_int8_4b.bmodel | 3.14     | 2.73           | 9.26          | 2.10     |
| BM1684X SoC | yolox_opencv.py  | yolox_s_fp32_1b.bmodel | 13.87    | 3.40           | 44.02         | 2.84     |
| BM1684X SoC | yolox_opencv.py  | yolox_s_fp16_1b.bmodel | 13.88    | 3.24           | 22.04         | 2.84     |
| BM1684X SoC | yolox_opencv.py  | yolox_s_int8_1b.bmodel | 13.98    | 3.49           | 20.94         | 2.82     |
| BM1684X SoC | yolox_opencv.py  | yolox_s_int8_4b.bmodel | 13.90    | 5.17           | 20.87         | 2.52     |
| BM1684X SoC | yolox_bmcv.py    | yolox_s_fp32_1b.bmodel | 3.22     | 2.40           | 30.33         | 2.86     |
| BM1684X SoC | yolox_bmcv.py    | yolox_s_fp16_1b.bmodel | 3.20     | 2.39           | 7.93          | 2.86     |
| BM1684X SoC | yolox_bmcv.py    | yolox_s_int8_1b.bmodel | 3.19     | 2.38           | 6.81          | 2.87     |
| BM1684X SoC | yolox_bmcv.py    | yolox_s_int8_4b.bmodel | 3.03     | 2.19           | 6.18          | 2.51     |
| BM1684X SoC | yolox_bmcv.soc   | yolox_s_fp32_1b.bmodel | 4.51     | 0.75           | 27.86         | 2.72     |
| BM1684X SoC | yolox_bmcv.soc   | yolox_s_fp16_1b.bmodel | 4.55     | 0.75           | 5.49          | 2.75     |
| BM1684X SoC | yolox_bmcv.soc   | yolox_s_int8_1b.bmodel | 4.51     | 0.75           | 4.35          | 2.74     |
| BM1684X SoC | yolox_bmcv.soc   | yolox_s_int8_4b.bmodel | 4.28     | 0.72           | 4.26          | 2.73     |
| BM1684X SoC | yolox_sail.soc   | yolox_s_fp32_1b.bmodel | 2.92     | 2.72           | 28.29         | 2.12     |
| BM1684X SoC | yolox_sail.soc   | yolox_s_fp16_1b.bmodel | 2.88     | 2.71           | 5.91          | 2.10     |
| BM1684X SoC | yolox_sail.soc   | yolox_s_int8_1b.bmodel | 2.81     | 2.72           | 4.76          | 2.10     |
| BM1684X SoC | yolox_sail.soc   | yolox_s_int8_4b.bmodel | 2.67     | 2.63           | 4.55          | 2.12     |
| BM1688 SoC  | yolox_opencv.py  | yolox_s_fp32_1b.bmodel | 21.62    | 4.17           | 174.24        | 3.98     |
| BM1688 SoC  | yolox_opencv.py  | yolox_s_fp16_1b.bmodel | 20.42    | 4.09           | 54.62         | 3.98     |
| BM1688 SoC  | yolox_opencv.py  | yolox_s_int8_1b.bmodel | 20.19    | 4.14           | 40.05         | 3.97     |
| BM1688 SoC  | yolox_bmcv.py    | yolox_s_fp32_1b.bmodel | 4.68     | 5.18           | 157.73        | 4.01     |
| BM1688 SoC  | yolox_bmcv.py    | yolox_s_fp16_1b.bmodel | 5.16     | 5.25           | 38.22         | 4.03     |
| BM1688 SoC  | yolox_bmcv.py    | yolox_s_int8_1b.bmodel | 4.53     | 5.18           | 23.63         | 3.99     |
| BM1688 SoC  | yolox_bmcv.soc   | yolox_s_fp32_1b.bmodel | 5.84     | 1.96           | 154.59        | 3.76     |
| BM1688 SoC  | yolox_bmcv.soc   | yolox_s_fp16_1b.bmodel | 5.83     | 1.94           | 35.05         | 3.75     |
| BM1688 SoC  | yolox_bmcv.soc   | yolox_s_int8_1b.bmodel | 5.78     | 1.95           | 20.42         | 3.74     |
| BM1688 SoC  | yolox_sail.soc   | yolox_s_fp32_1b.bmodel | 3.94     | 5.22           | 155.22        | 2.91     | 
| BM1688 SoC  | yolox_sail.soc   | yolox_s_fp16_1b.bmodel | 3.92     | 5.21           | 5.65          | 2.91     |
| BM1688 SoC  | yolox_sail.soc   | yolox_s_int8_1b.bmodel | 4.01     | 5.23           | 21.04         | 2.91     |

> **测试说明**：  
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. BM1684 SoC的测试平台为标准版SE5，BM1684X SoC的测试平台为标准版SE7；
> 4. BM1684/1684X SoC的主控处理器均为8核 ARM A53 42320 DMIPS @2.3GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 5. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异，不同的阈值对后处理时间影响较大；
> 6. BM1688双核模型性能与单核模型相比，推理时间不同，其他部分基本一致，推理性能区别请参考[7.1小节](#71-bmrt_test)测试数据；
> 7. `yolox_opencv.py`的decode_time基于公版opencv。

## 8. FAQ
[常见问题解答](../../docs/FAQ.md)

## 9. 致谢
* 感谢 “灵耘致新” 对YOLOX的python例程的优化
