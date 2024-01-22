# CenterNet

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

CenterNet 是一种 anchor-free 的目标检测网络，不仅可以用于目标检测，还可以用于其他的一些任务，如姿态识别或者 3D 目标检测等等。

**文档:** [CenterNet论文](https://arxiv.org/pdf/1904.07850.pdf)

**参考repo:** [CenterNet](https://github.com/xingyizhou/CenterNet)

## 2. 特性
* 支持BM1688(SoC)、BM1684X(x86 PCIe、SoC)和BM1684(x86 PCIe、SoC、arm PCIe)
* 支持FP32、FP16(BM1688/BM1684X)、INT8模型编译和推理
* 支持基于BMCV预处理和sail的C++推理
* 支持基于BMCV和opencv预处理的Python推理
* 支持单batch和多batch模型推理
* 支持图片测试

## 3. 准备模型与数据
如果您使用BM1684，建议使用TPU-NNTC编译BModel，Pytorch模型在编译前要导出成torchscript模型或onnx模型；如果您使用BM1684X，建议使用TPU-MLIR编译BModel，Pytorch模型在编译前要导出成onnx模型。具体可参考[centernet模型导出](./docs/CenterNet_Export_Guide.md)。

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
│   ├── centernet_fp32_1b.bmodel   # 使用TPU-NNTC编译，用于BM1684的FP32 BModel，batch_size=1
│   ├── centernet_int8_1b.bmodel   # 使用TPU-NNTC编译，用于BM1684的INT8 BModel，batch_size=1
│   └── centernet_int8_4b.bmodel   # 使用TPU-NNTC编译，用于BM1684的INT8 BModel，batch_size=4
├── BM1684X
│   ├── centernet_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=1
│   ├── centernet_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=1
│   ├── centernet_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=1
│   └── centernet_int8_4b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=4
├── BM1688
│   ├── centernet_fp32_1b.bmodel         # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1，num_core=1
│   ├── centernet_fp16_1b.bmodel         # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1，num_core=1
│   ├── centernet_int8_1b.bmodel         # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=1，num_core=1
│   ├── centernet_int8_4b.bmodel         # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=4，num_core=1
│   ├── centernet_fp32_1b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1，num_core=2
│   ├── centernet_fp16_1b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1，num_core=2
│   ├── centernet_int8_1b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=1，num_core=2
│   └── centernet_int8_4b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=4，num_core=2
│── torch
│   ├── ctdet_coco_dlav0_1x.pth
│   └── ctdet_coco_dlav0_1x.torchscript.pt   # trace后的torchscript模型
└── onnx
    ├── centernet_1b.onnx
    ├── centernet_4b.onnx         # 导出的onnx动态模型 
    └── dlav0_qtable              # mlir量化需要的保留精度的层      
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
导出的模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。建议使用TPU-MLIR编译BModel。

模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index/material/31/all.html)相应版本的SDK中获取)。

- 生成FP32 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684/BM1684X/BM1688**），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684 #bm1684x/bm1688
```

​执行上述命令会在`models/BM1684`等文件夹下生成`centernet_fp32_1b.bmodel`文件，即转换好的FP32 BModel。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688**），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x #bm1688
```

​执行上述命令会在`models/BM1684X/`等文件夹下生成`centernet_fp16_1b.bmodel`文件，即转换好的FP16 BModel。

- 生成INT8 BModel

​本例程在`scripts`目录下提供了量化INT8 BModel的脚本，请注意修改`gen_int8bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台（**支持BM1684/BM1684X/BM1688**），如：

```shell
./scripts/gen_int8bmodel_mlir.sh bm1684 #bm1684x/bm1688
```

​上述脚本会在`models/BM1684`等文件夹下生成`centernet_int8_1b.bmodel`等文件，即转换好的INT8 BModel。

**注：int8模型有提供qtable，即保留精度不量化的层，一般指定centernet的最后几层，包括relu、conv等，详情可参考`models/onnx/dlav0_qtable`中内容，按照实际模型结构的层名添加。**


## 5. 例程测试
- [C++例程](./cpp/README.md)
- [Python例程](./python/README.md)

## 6. 精度测试
### 6.1 测试方法

首先，参考[C++例程](cpp/README.md#32-测试图片)或[Python例程](python/README.md#22-测试图片)推理要测试的数据集，生成预测的json文件，注意修改数据集(datasets/coco/val2017_1000)和相关参数(conf_thresh=0.35)。  
然后，使用`tools`目录下的`eval_coco.py`脚本，将测试生成的json文件与测试集标签json文件进行对比，计算出目标检测的评价指标，命令如下：
```bash
# 安装pycocotools，若已安装请跳过
pip3 install pycocotools
# 请根据实际情况修改程序路径和json文件路径
python3 tools/eval_coco.py --gt_path datasets/coco/instances_val2017_1000.json --result_json results/centernet_fp32_1b.bmodel_val2017_1000_bmcv_python_result.json
```
### 6.2 测试结果
在coco2017val_1000数据集上，精度测试结果如下：
|   测试平台    |      测试程序       |              测试模型      |AP@IoU=0.5:0.95|AP@IoU=0.5|
| ------------ | ----------------   | ------------------------- | ------------- | -------- |
| BM1684 PCIe  | centernet_opencv.py | centernet_fp32_1b.bmodel | 0.302         | 0.487    |
| BM1684 PCIe  | centernet_opencv.py | centernet_int8_1b.bmodel | 0.294         | 0.484    |
| BM1684 PCIe  | centernet_bmcv.py   | centernet_fp32_1b.bmodel | 0.255         | 0.409    |
| BM1684 PCIe  | centernet_bmcv.py   | centernet_int8_1b.bmodel | 0.249         | 0.405    |
| BM1684 PCIe  | centernet_sail.pcie | centernet_fp32_1b.bmodel | 0.297         | 0.481    |
| BM1684 PCIe  | centernet_sail.pcie | centernet_int8_1b.bmodel | 0.290         | 0.476    |
| BM1684 PCIe  | centernet_bmcv.pcie | centernet_fp32_1b.bmodel | 0.268         | 0.430    |
| BM1684 PCIe  | centernet_bmcv.pcie | centernet_int8_1b.bmodel | 0.258         | 0.422    |
| BM1684X PCIe | centernet_opencv.py | centernet_fp32_1b.bmodel | 0.302         | 0.487    |
| BM1684X PCIe | centernet_opencv.py | centernet_fp16_1b.bmodel | 0.302         | 0.487    |
| BM1684X PCIe | centernet_opencv.py | centernet_int8_1b.bmodel | 0.299         | 0.485    |
| BM1684X PCIe | centernet_bmcv.py   | centernet_fp32_1b.bmodel | 0.258         | 0.421    |
| BM1684X PCIe | centernet_bmcv.py   | centernet_fp16_1b.bmodel | 0.258         | 0.422    |
| BM1684X PCIe | centernet_bmcv.py   | centernet_int8_1b.bmodel | 0.257         | 0.419    |
| BM1684X PCIe | centernet_sail.pcie | centernet_fp32_1b.bmodel | 0.296         | 0.480    |
| BM1684X PCIe | centernet_sail.pcie | centernet_fp16_1b.bmodel | 0.296         | 0.480    |
| BM1684X PCIe | centernet_sail.pcie | centernet_int8_1b.bmodel | 0.294         | 0.477    |
| BM1684X PCIe | centernet_bmcv.pcie | centernet_fp32_1b.bmodel | 0.268         | 0.429    |
| BM1684X PCIe | centernet_bmcv.pcie | centernet_fp16_1b.bmodel | 0.268         | 0.430    |
| BM1684X PCIe | centernet_bmcv.pcie | centernet_int8_1b.bmodel | 0.264         | 0.425    |
| BM1688 SoC   | centernet_opencv.py | centernet_fp32_1b.bmodel | 0.302         | 0.487    |
| BM1688 SoC   | centernet_opencv.py | centernet_fp16_1b.bmodel | 0.302         | 0.487    |
| BM1688 SoC   | centernet_opencv.py | centernet_int8_1b.bmodel | 0.298         | 0.489    |
| BM1688 SoC   | centernet_bmcv.py   | centernet_fp32_1b.bmodel | 0.258         | 0.421    |
| BM1688 SoC   | centernet_bmcv.py   | centernet_fp16_1b.bmodel | 0.258         | 0.422    |
| BM1688 SoC   | centernet_bmcv.py   | centernet_int8_1b.bmodel | 0.255         | 0.414    |
| BM1688 SoC   | centernet_bmcv.soc  | centernet_fp32_1b.bmodel | 0.268         | 0.429    |
| BM1688 SoC   | centernet_bmcv.soc  | centernet_fp16_1b.bmodel | 0.268         | 0.430    |
| BM1688 SoC   | centernet_bmcv.soc  | centernet_int8_1b.bmodel | 0.265         | 0.426    |
| BM1688 SoC   | centernet_sail.soc  | centernet_fp32_1b.bmodel | 0.296         | 0.480    |
| BM1688 SoC   | centernet_sail.soc  | centernet_fp16_1b.bmodel | 0.296         | 0.480    |
| BM1688 SoC   | centernet_sail.soc  | centernet_int8_1b.bmodel | 0.289         | 0.467    |

> **测试说明**：  
> 1. SoC和PCIe的模型精度一致，int8 1b和4b的精度一致；
> 2. AP@IoU=0.5:0.95为area=all对应的指标；
> 3. bmcv的精度略低于其他，主要是预处理的一些方法与源码有差异；
> 4. 由于sdk版本之间可能存在差异，实际运行结果与本表有<0.01的精度误差是正常的；
> 5. BM1688 num_core=2的模型与num_core=1的模型精度基本一致。

## 7. 性能测试
### 7.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684/centernet_fp32_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

|               测试模型                  | calculate time(ms) |
| ---------------------------------------| ----------------- |
| BM1684/centernet_fp32_1b.bmodel        | 46.2              |
| BM1684/centernet_int8_1b.bmodel        | 22.5              |
| BM1684/centernet_int8_4b.bmodel        | 8.1               |
| BM1684X/centernet_fp32_1b.bmodel       | 55.1              |
| BM1684X/centernet_fp16_1b.bmodel       | 9.2               |
| BM1684X/centernet_int8_1b.bmodel       | 4.4               |
| BM1684X/centernet_int8_4b.bmodel       | 4.0               |
| BM1688/centernet_fp32_1b.bmodel        | 333.2             |
| BM1688/centernet_fp16_1b.bmodel        | 48.5              |
| BM1688/centernet_int8_1b.bmodel        | 21.9              |
| BM1688/centernet_int8_4b.bmodel        | 20.5              |
| BM1688/centernet_fp32_1b_2core.bmodel  | 248.8             |
| BM1688/centernet_fp16_1b_2core.bmodel  | 32.9              |
| BM1688/centernet_int8_1b_2core.bmodel  | 15.1              |
| BM1688/centernet_int8_4b_2core.bmodel  | 11.5              |

> **测试说明**：  
> 1. 性能测试结果具有一定的波动性；
> 2. `calculate time`已折算为平均每张图片的推理时间；
> 3. SoC和PCIe的测试结果基本一致。

### 7.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++和Python例程打印的时间已经折算为单张图片的处理时间。

在不同的测试平台上，使用不同的例程、模型测试`datasets/coco/val2017_1000`，设置参数`--conf_thresh=0.35`，性能测试结果如下：
|    测试平台  |     测试程序           |             测试模型       |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ----------------      | ---------------------------| --------  | ---------     | ---------    | ---------    |
| BM1684 SoC  | centernet_opencv.py   | centernet_fp32_1b.bmodel   | 15.33     | 40.37         | 59.70        | 822.85       |
| BM1684 SoC  | centernet_opencv.py   | centernet_int8_1b.bmodel   | 15.46     | 40.83         | 44.46        | 812.17       |
| BM1684 SoC  | centernet_opencv.py   | centernet_int8_4b.bmodel   | 15.32     | 39.22         | 28.77        | 842.73       |
| BM1684 SoC  | centernet_bmcv.py     | centernet_fp32_1b.bmodel   | 3.03      | 2.61          | 50.53        | 820.98       |
| BM1684 SoC  | centernet_bmcv.py     | centernet_int8_1b.bmodel   | 3.04      | 2.25          | 26.18        | 827.04       |
| BM1684 SoC  | centernet_bmcv.py     | centernet_int8_4b.bmodel   | 2.89      | 2.11          | 11.79        | 851.17       |
| BM1684 SoC  | centernet_sail.soc    | centernet_fp32_1b.bmodel   | 3.48      | 1.87          | 46.94        | 1350.56      |
| BM1684 SoC  | centernet_sail.soc    | centernet_int8_1b.bmodel   | 3.53      | 1.29          | 23.11        | 1352.22      |
| BM1684 SoC  | centernet_sail.soc    | centernet_int8_4b.bmodel   | 3.24      | 1.01          | 8.46         | 1352.32      |
| BM1684 SoC  | centernet_bmcv.soc    | centernet_fp32_1b.bmodel   | 5.46      | 1.47          | 46.27        | 1179.17      |
| BM1684 SoC  | centernet_bmcv.soc    | centernet_int8_1b.bmodel   | 5.42      | 1.46          | 22.51        | 1180.33      |
| BM1684 SoC  | centernet_bmcv.soc    | centernet_int8_4b.bmodel   | 5.37      | 1.50          | 7.9          | 1189.37      |
| BM1684X SoC | centernet_opencv.py   | centernet_fp32_1b.bmodel   | 15.31     | 40.02         | 71.32        | 817.13       |
| BM1684X SoC | centernet_opencv.py   | centernet_int8_1b.bmodel   | 15.37     | 40.12         | 19.9         | 802.71       |
| BM1684X SoC | centernet_opencv.py   | centernet_int8_4b.bmodel   | 15.23     | 38.37         | 18.32        | 836.56       |
| BM1684X SoC | centernet_bmcv.py     | centernet_fp32_1b.bmodel   | 2.98      | 1.99          | 60.61        | 817.42       |
| BM1684X SoC | centernet_bmcv.py     | centernet_int8_1b.bmodel   | 2.99      | 1.99          | 8.97         | 801.77       |
| BM1684X SoC | centernet_bmcv.py     | centernet_int8_4b.bmodel   | 2.80      | 1.84          | 8.15         | 841.5        |
| BM1684X SoC | centernet_sail.soc    | centernet_fp32_1b.bmodel   | 2.9       | 1.63          | 57.1         | 1356.13      |
| BM1684X SoC | centernet_sail.soc    | centernet_int8_1b.bmodel   | 2.92      | 1.63          | 5.6          | 1357.22      |
| BM1684X SoC | centernet_sail.soc    | centernet_int8_4b.bmodel   | 2.64      | 1.57          | 5.08         | 1358.17      |
| BM1684X SoC | centernet_bmcv.soc    | centernet_fp32_1b.bmodel   | 4.77      | 0.75          | 56.53        | 1184.43      |
| BM1684X SoC | centernet_bmcv.soc    | centernet_int8_1b.bmodel   | 4.8       | 0.75          | 4.99         | 1185.35      |
| BM1684X SoC | centernet_bmcv.soc    | centernet_int8_4b.bmodel   | 4.47      | 0.68          | 4.52         | 1185.1       |
| BM1688 SoC  | centernet_opencv.py   | centernet_fp32_1b.bmodel   | 19.49     | 52.54         | 349.78       | 1116.06      |
| BM1688 SoC  | centernet_opencv.py   | centernet_fp16_1b.bmodel   | 19.39     | 52.20         | 64.80        | 1093.04      |
| BM1688 SoC  | centernet_opencv.py   | centernet_int8_1b.bmodel   | 19.47     | 52.17         | 38.51        | 1083.13      |
| BM1688 SoC  | centernet_opencv.py   | centernet_int8_4b.bmodel   | 19.39     | 50.10         | 36.29        | 1113.67      |
| BM1688 SoC  | centernet_bmcv.py     | centernet_fp32_1b.bmodel   | 4.58      | 4.52          | 337.37       | 1093.85      |
| BM1688 SoC  | centernet_bmcv.py     | centernet_fp16_1b.bmodel   | 4.53      | 4.51          | 52.45        | 1087.40      |
| BM1688 SoC  | centernet_bmcv.py     | centernet_int8_1b.bmodel   | 4.57      | 4.51          | 25.96        | 1063.44      |
| BM1688 SoC  | centernet_bmcv.py     | centernet_int8_4b.bmodel   | 4.29      | 4.20          | 24.56        | 1129.51      |
| BM1688 SoC  | centernet_bmcv.soc    | centernet_fp32_1b.bmodel   | 6.10      | 1.70          | 332.19       | 1620.93      |
| BM1688 SoC  | centernet_bmcv.soc    | centernet_fp16_1b.bmodel   | 6.11      | 1.71          | 47.42        | 1620.83      |
| BM1688 SoC  | centernet_bmcv.soc    | centernet_int8_1b.bmodel   | 6.13      | 1.70          | 20.83        | 1622.64      |
| BM1688 SoC  | centernet_bmcv.soc    | centernet_int8_4b.bmodel   | 5.85      | 1.55          | 20.03        | 1623.05      |
| BM1688 SoC  | centernet_sail.soc    | centernet_fp32_1b.bmodel   | 4.18      | 3.19          | 333.22       | 1859.35      |
| BM1688 SoC  | centernet_sail.soc    | centernet_fp16_1b.bmodel   | 4.21      | 3.19          | 48.40        | 1859.22      |
| BM1688 SoC  | centernet_sail.soc    | centernet_int8_1b.bmodel   | 4.21      | 3.18          | 21.79        | 1861.07      |
| BM1688 SoC  | centernet_sail.soc    | centernet_int8_4b.bmodel   | 3.85      | 3.04          | 21.08        | 1861.63      |

> **测试说明**：  
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. BM1684/1684X SoC的主控处理器均为8核 ARM A53 42320 DMIPS @2.3GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异； 

## 8. FAQ
请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。