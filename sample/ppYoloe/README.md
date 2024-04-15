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
* 支持BM1688/CV186X(SoC)、BM1684X(x86 PCIe、SoC)和BM1684(x86 PCIe、SoC、arm PCIe)
* 支持FP32、FP16(BM1684X/BM1688/CV186X)模型编译和推理
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
├── BM1688
│   ├── ppyoloe_fp32_1b.bmodel        # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1
│   ├── ppyoloe_fp16_1b.bmodel        # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1
│   ├── ppyoloe_fp32_1b_2core.bmodel  # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1, num_core=2
│   └── ppyoloe_fp16_1b_2core.bmodel  # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1, num_core=2
├── CV186X
│   ├── ppyoloe_fp32_1b.bmodel        # 使用TPU-MLIR编译，用于CV186X的FP32 BModel，batch_size=1
│   └── ppyoloe_fp16_1b.bmodel        # 使用TPU-MLIR编译，用于CV186X的FP16 BModel，batch_size=1
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

​本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684/BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684 #bm1684x/bm1688/cv186x
```

​执行上述命令会在`models/BM1684`等文件夹下生成`ppyoloe_fp32_1b.bmodel`文件，即转换好的FP32 BModel。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x #bm1688/cv186x
```

​执行上述命令会在`models/BM1684X/`等文件夹下生成`ppyoloe_fp16_1b.bmodel`文件，即转换好的FP16 BModel。

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

在`datasets/coco/val2017_1000`数据集上，精度测试结果如下：

|   测试平台    |      测试程序     |        测试模型        | AP@IoU=0.5:0.95 | AP@IoU=0.5 |
| ------------ | ---------------- | ---------------------- | --------------- | ---------- |
| SE5-16       | ppyoloe_opencv.py  | ppyoloe_fp32_1b.bmodel       |    0.377 |    0.508 |
| SE5-16       | ppyoloe_bmcv.py    | ppyoloe_fp32_1b.bmodel       |    0.381 |    0.513 |
| SE5-16       | ppyoloe_bmcv.soc   | ppyoloe_fp32_1b.bmodel       |    0.378 |    0.510 |
| SE5-16       | ppyoloe_sail.soc   | ppyoloe_fp32_1b.bmodel       |    0.378 |    0.510 |
| SE7-32       | ppyoloe_opencv.py  | ppyoloe_fp32_1b.bmodel       |    0.377 |    0.508 |
| SE7-32       | ppyoloe_opencv.py  | ppyoloe_fp16_1b.bmodel       |    0.377 |    0.508 |
| SE7-32       | ppyoloe_bmcv.py    | ppyoloe_fp32_1b.bmodel       |    0.380 |    0.513 |
| SE7-32       | ppyoloe_bmcv.py    | ppyoloe_fp16_1b.bmodel       |    0.380 |    0.513 |
| SE7-32       | ppyoloe_bmcv.soc   | ppyoloe_fp32_1b.bmodel       |    0.379 |    0.510 |
| SE7-32       | ppyoloe_bmcv.soc   | ppyoloe_fp16_1b.bmodel       |    0.378 |    0.510 |
| SE7-32       | ppyoloe_sail.soc   | ppyoloe_fp32_1b.bmodel       |    0.379 |    0.510 |
| SE7-32       | ppyoloe_sail.soc   | ppyoloe_fp16_1b.bmodel       |    0.378 |    0.510 |
| SE9-16       | ppyoloe_opencv.py  | ppyoloe_fp32_1b.bmodel       |    0.377 |    0.508 |
| SE9-16       | ppyoloe_opencv.py  | ppyoloe_fp16_1b.bmodel       |    0.376 |    0.508 |
| SE9-16       | ppyoloe_bmcv.py    | ppyoloe_fp32_1b.bmodel       |    0.381 |    0.513 |
| SE9-16       | ppyoloe_bmcv.py    | ppyoloe_fp16_1b.bmodel       |    0.380 |    0.514 |
| SE9-16       | ppyoloe_bmcv.soc   | ppyoloe_fp32_1b.bmodel       |    0.379 |    0.510 |
| SE9-16       | ppyoloe_bmcv.soc   | ppyoloe_fp16_1b.bmodel       |    0.378 |    0.510 |
| SE9-16       | ppyoloe_sail.soc   | ppyoloe_fp32_1b.bmodel       |    0.379 |    0.510 |
| SE9-16       | ppyoloe_sail.soc   | ppyoloe_fp16_1b.bmodel       |    0.378 |    0.510 |
| SE9-16       | ppyoloe_opencv.py  | ppyoloe_fp32_1b_2core.bmodel |    0.377 |    0.508 |
| SE9-16       | ppyoloe_opencv.py  | ppyoloe_fp16_1b_2core.bmodel |    0.376 |    0.508 |
| SE9-16       | ppyoloe_bmcv.py    | ppyoloe_fp32_1b_2core.bmodel |    0.381 |    0.513 |
| SE9-16       | ppyoloe_bmcv.py    | ppyoloe_fp16_1b_2core.bmodel |    0.380 |    0.514 |
| SE9-16       | ppyoloe_bmcv.soc   | ppyoloe_fp32_1b_2core.bmodel |    0.379 |    0.510 |
| SE9-16       | ppyoloe_bmcv.soc   | ppyoloe_fp16_1b_2core.bmodel |    0.378 |    0.510 |
| SE9-16       | ppyoloe_sail.soc   | ppyoloe_fp32_1b_2core.bmodel |    0.379 |    0.510 |
| SE9-16       | ppyoloe_sail.soc   | ppyoloe_fp16_1b_2core.bmodel |    0.378 |    0.510 |
| SE9-8        | ppyoloe_opencv.py  | ppyoloe_fp32_1b.bmodel       |    0.377 |    0.508 |
| SE9-8        | ppyoloe_opencv.py  | ppyoloe_fp16_1b.bmodel       |    0.376 |    0.508 |
| SE9-8        | ppyoloe_bmcv.py    | ppyoloe_fp32_1b.bmodel       |    0.381 |    0.513 |
| SE9-8        | ppyoloe_bmcv.py    | ppyoloe_fp16_1b.bmodel       |    0.380 |    0.514 |
| SE9-8        | ppyoloe_bmcv.soc   | ppyoloe_fp32_1b.bmodel       |    0.379 |    0.510 |
| SE9-8        | ppyoloe_bmcv.soc   | ppyoloe_fp16_1b.bmodel       |    0.378 |    0.510 |
| SE9-8        | ppyoloe_sail.soc   | ppyoloe_fp32_1b.bmodel       |    0.379 |    0.510 |
| SE9-8        | ppyoloe_sail.soc   | ppyoloe_fp16_1b.bmodel       |    0.378 |    0.510 |

> **测试说明**：  
> 1. 由于sdk版本之间可能存在差异，实际运行结果与本表有<0.01的精度误差是正常的；
> 2. AP@IoU=0.5:0.95为area=all对应的指标；
> 3. 在搭载了相同TPU和SOPHONSDK的PCIe或SoC平台上，相同程序的精度一致，SE5系列对应BM1684，SE7系列对应BM1684X，SE9系列中，SE9-16对应BM1688，SE9-8对应CV186X；

## 7. 性能测试
### 7.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684/ppyoloe_fp32_1b.bmodel
```
测试结果中的`calculate time()`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

| 测试模型                       | calculate time(ms) |
| -------------------                |  -------------- |
| BM1684/ppyoloe_fp32_1b.bmodel      |          30.77  |
| BM1684X/ppyoloe_fp32_1b.bmodel     |          27.38  |
| BM1684X/ppyoloe_fp16_1b.bmodel     |           6.85  |
| BM1688/ppyoloe_fp32_1b.bmodel      |         119.37  |
| BM1688/ppyoloe_fp16_1b.bmodel      |          31.43  |
| BM1688/ppyoloe_fp32_1b_2core.bmodel|          76.65  |
| BM1688/ppyoloe_fp16_1b_2core.bmodel|          21.21  |
| CV186X/ppyoloe_fp32_1b.bmodel      |         122.38  |
| CV186X/ppyoloe_fp16_1b.bmodel      |          34.91  |

> **测试说明**：  
> 1. 性能测试结果具有一定的波动性；
> 2. `calculate time`已折算为平均每张图片的推理时间；
> 3. SoC和PCIe的测试结果基本一致。

### 7.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++和Python例程打印的时间已经折算为单张图片的处理时间。


在不同的测试平台上，使用不同的例程、模型测试`datasets/val2017_1000`，conf_thresh=0.4，nms_thresh=0.6，性能测试结果如下：
|    测试平台  |     测试程序      |     测试模型          |decode_time|preprocess_time|inference_time|postprocess_time|
| ----------- | ---------------- | ---------------------- | -------- | -------------- | ---------     | --------- |
|   SE5-16    | ppyoloe_opencv.py |      ppyoloe_fp32_1b.bmodel       |      15.02      |      43.29      |      45.16      |      12.46      |
|   SE5-16    |  ppyoloe_bmcv.py  |      ppyoloe_fp32_1b.bmodel       |      3.76       |      3.63       |      34.07      |      12.64      |
|   SE5-16    | ppyoloe_bmcv.soc  |      ppyoloe_fp32_1b.bmodel       |      4.84       |      1.08       |      30.71      |      8.57       |
|   SE5-16    | ppyoloe_sail.soc  |      ppyoloe_fp32_1b.bmodel       |      3.21       |      4.14       |      31.11      |      7.99       |
|   SE7-32    | ppyoloe_opencv.py |      ppyoloe_fp32_1b.bmodel       |      16.16      |      40.55      |      44.29      |      12.95      |
|   SE7-32    | ppyoloe_opencv.py |      ppyoloe_fp16_1b.bmodel       |      15.13      |      40.54      |      23.72      |      12.88      |
|   SE7-32    |  ppyoloe_bmcv.py  |      ppyoloe_fp32_1b.bmodel       |      3.30       |      2.81       |      30.95      |      13.42      |
|   SE7-32    |  ppyoloe_bmcv.py  |      ppyoloe_fp16_1b.bmodel       |      3.26       |      2.79       |      10.40      |      13.44      |
|   SE7-32    | ppyoloe_bmcv.soc  |      ppyoloe_fp32_1b.bmodel       |      4.34       |      0.99       |      27.35      |      8.70       |
|   SE7-32    | ppyoloe_bmcv.soc  |      ppyoloe_fp16_1b.bmodel       |      4.35       |      0.99       |      6.79       |      8.72       |
|   SE7-32    | ppyoloe_sail.soc  |      ppyoloe_fp32_1b.bmodel       |      2.71       |      3.33       |      27.76      |      8.10       |
|   SE7-32    | ppyoloe_sail.soc  |      ppyoloe_fp16_1b.bmodel       |      2.71       |      3.33       |      7.22       |      8.11       |
|   SE9-16    | ppyoloe_opencv.py |      ppyoloe_fp32_1b.bmodel       |      23.48      |      55.35      |     139.76      |      17.91      |
|   SE9-16    | ppyoloe_opencv.py |      ppyoloe_fp16_1b.bmodel       |      19.24      |      55.54      |      51.71      |      17.77      |
|   SE9-16    |  ppyoloe_bmcv.py  |      ppyoloe_fp32_1b.bmodel       |      4.59       |      5.44       |     124.17      |      17.67      |
|   SE9-16    |  ppyoloe_bmcv.py  |      ppyoloe_fp16_1b.bmodel       |      4.61       |      5.41       |      36.18      |      17.95      |
|   SE9-16    | ppyoloe_bmcv.soc  |      ppyoloe_fp32_1b.bmodel       |      5.75       |      2.24       |     119.31      |      12.15      |
|   SE9-16    | ppyoloe_bmcv.soc  |      ppyoloe_fp16_1b.bmodel       |      5.80       |      2.23       |      31.36      |      12.14      |
|   SE9-16    | ppyoloe_sail.soc  |      ppyoloe_fp32_1b.bmodel       |      3.80       |      6.12       |     120.29      |      11.33      |
|   SE9-16    | ppyoloe_sail.soc  |      ppyoloe_fp16_1b.bmodel       |      3.75       |      6.11       |      32.33      |      11.32      |
|   SE9-16    | ppyoloe_opencv.py |   ppyoloe_fp32_1b_2core.bmodel    |      19.29      |      55.28      |      96.96      |      17.91      |
|   SE9-16    | ppyoloe_opencv.py |   ppyoloe_fp16_1b_2core.bmodel    |      19.16      |      55.66      |      41.51      |      17.78      |
|   SE9-16    |  ppyoloe_bmcv.py  |   ppyoloe_fp32_1b_2core.bmodel    |      4.54       |      5.41       |      81.45      |      17.71      |
|   SE9-16    |  ppyoloe_bmcv.py  |   ppyoloe_fp16_1b_2core.bmodel    |      4.54       |      5.39       |      25.95      |      17.74      |
|   SE9-16    | ppyoloe_bmcv.soc  |   ppyoloe_fp32_1b_2core.bmodel    |      5.83       |      2.23       |      76.57      |      12.14      |
|   SE9-16    | ppyoloe_bmcv.soc  |   ppyoloe_fp16_1b_2core.bmodel    |      5.80       |      2.23       |      21.16      |      12.15      |
|   SE9-16    | ppyoloe_sail.soc  |   ppyoloe_fp32_1b_2core.bmodel    |      3.81       |      6.11       |      77.55      |      11.32      |
|   SE9-16    | ppyoloe_sail.soc  |   ppyoloe_fp16_1b_2core.bmodel    |      3.77       |      6.10       |      22.11      |      11.33      |
|    SE9-8    | ppyoloe_opencv.py |      ppyoloe_fp32_1b.bmodel       |      24.17      |      55.57      |     142.53      |      17.66      |
|    SE9-8    | ppyoloe_opencv.py |      ppyoloe_fp16_1b.bmodel       |      20.70      |      55.99      |      55.20      |      17.54      |
|    SE9-8    |  ppyoloe_bmcv.py  |      ppyoloe_fp32_1b.bmodel       |      7.66       |      5.63       |     127.22      |      17.71      |
|    SE9-8    |  ppyoloe_bmcv.py  |      ppyoloe_fp16_1b.bmodel       |      4.45       |      5.59       |      39.63      |      17.73      |
|    SE9-8    | ppyoloe_bmcv.soc  |      ppyoloe_fp32_1b.bmodel       |      5.61       |      2.59       |     122.30      |      12.15      |
|    SE9-8    | ppyoloe_bmcv.soc  |      ppyoloe_fp16_1b.bmodel       |      5.64       |      2.59       |      34.85      |      12.14      |
|    SE9-8    | ppyoloe_sail.soc  |      ppyoloe_fp32_1b.bmodel       |      3.67       |      6.34       |     123.30      |      11.33      |
|    SE9-8    | ppyoloe_sail.soc  |      ppyoloe_fp16_1b.bmodel       |      3.64       |      6.32       |      35.81      |      11.33      |

> **测试说明**：  
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. SE5-16/SE7-32的主控处理器均为8核CA53@2.3GHz，SE9-16为8核CA53@1.6GHz，SE9-8为6核CA53@1.6GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异，不同的阈值对后处理时间影响较大。 

## 8. FAQ
[常见问题解答](../../docs/FAQ.md)
