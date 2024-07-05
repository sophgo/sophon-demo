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

ppyolov3 是百度提出的一种基于YOLOv3和一些几乎不增加推理代价的tricks改造而来的检测器，达到了不错的速度-精度权衡。

**论文地址** (https://arxiv.org/pdf/2007.12099.pdf)

**官方源码地址** (https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/ppyolo)

## 2. 特性
* 支持BM1688/CV186X(SoC)、BM1684X(x86 PCIe、SoC)和BM1684(x86 PCIe、SoC、arm PCIe)
* 支持FP32、FP16(BM1684X/BM1688/CV186X)、INT8模型编译和推理
* 支持基于BMCV预处理的C++推理
* 支持基于BMCV和opencv预处理的Python推理
* 支持图片和视频测试

## 3. 准备模型与数据
百度的飞桨PaddlePaddle模型权重来源于[yolov3.pdparams](https://paddledet.bj.bcebos.com/models/yolov3_darknet53_270e_coco.pdparams)，配置文件来源于[yolov3.yml](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.6/configs/yolov3/yolov3_darknet53_270e_coco.yml)

建议使用TPU-MLIR编译BModel，百度的飞桨PaddlePaddle模型在编译前要导出成onnx模型。导出可参考：https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.7/deploy/EXPORT_ONNX_MODEL.md

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
│   └── ppyolov3_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684的INT8 BModel，batch_size=1
├── BM1684X
│   ├── ppyolov3_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=1
│   ├── ppyolov3_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=1
│   └── ppyolov3_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=1
├── BM1684X
│   ├── ppyolov3_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=1
│   ├── ppyolov3_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=1
│   └── ppyolov3_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=1
├── BM1688
│   ├── ppyolov3_fp32_1b.bmodel         # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1，num_core=1
│   ├── ppyolov3_fp16_1b.bmodel         # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1，num_core=1
│   ├── ppyolov3_int8_1b.bmodel         # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=1，num_core=1
│   ├── ppyolov3_fp32_1b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1，num_core=2
│   ├── ppyolov3_fp16_1b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1，num_core=2
│   └── ppyolov3_int8_1b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=1，num_core=2
├── CV186X
│   ├── ppyolov3_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=1
│   ├── ppyolov3_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=1
│   └── ppyolov3_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=1
└── onnx
    └── ppyolov3_1b.onnx           # 导出的1batch onnx模型   
```
下载的数据包括：
```
./datasets
├── test                                      # 测试图片
├── coco.names                                # coco类别名文件
├── coco128                                   # coco128数据集，用于模型量化
└── coco                                      
    ├── val2017_1000                          # coco val2017_1000数据集：coco val2017中随机抽取的1000张样本
    └── instances_val2017_1000.json           # coco val2017_1000数据集标签文件，用于计算精度评价指标  
```

## 4. 模型编译
导出的模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。建议使用TPU-MLIR编译BModel。

模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的SDK中获取)。

- 生成FP32 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684/BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684 #bm1684x/bm1688/cv186x
```

​执行上述命令会在`models/BM1684`下生成`ppyolov3_fp32_1b.bmodel`文件，即转换好的FP32 BModel。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x #bm1688/cv186x
```

​执行上述命令会在`models/BM1684X/`下生成`ppyolov3_fp16_1b.bmodel`文件，即转换好的FP16 BModel。

- 生成INT8 BModel

​本例程在`scripts`目录下提供了量化INT8 BModel的脚本，请注意修改`gen_int8bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台（**支持BM1684/BM1684X/BM1688/CV186X**），如：

```shell
./scripts/gen_int8bmodel_mlir.sh bm1684 #bm1684x/bm1688/cv186x
```

​上述脚本会在`models/BM1684`下生成`ppyolov3_int8_1b.bmodel`等文件，即转换好的INT8 BModel。


## 5. 例程测试
- [C++例程](./cpp/README.md)
- [Python例程](./python/README.md)

## 6. 精度测试
### 6.1 测试方法

首先，参考[C++例程](cpp/README.md#32-测试图片)或[Python例程](python/README.md#22-测试图片)推理要测试的数据集，生成预测的json文件，注意修改数据集(datasets/coco/val2017_1000)和相关参数。  
然后，使用`tools`目录下的`eval_coco.py`脚本，将测试生成的json文件与测试集标签json文件进行对比，计算出目标检测的评价指标，命令如下：
```bash
# 安装pycocotools，若已安装请跳过
pip3 install pycocotools
# 请根据实际情况修改程序路径和json文件路径
python3 tools/eval_coco.py --gt_path datasets/coco/instances_val2017_1000.json --result_json results/ppyolov3_fp32_1b.bmodel_val2017_1000_bmcv_python_result.json
```
### 6.2 测试结果
在`datasets/coco/val2017_1000`数据集上，**推理时设置参数：--conf_thresh=0.001 --nms_thresh=0.6**，ppyolov3精度测试结果如下：
|   测试平台   |      测试程序      |         测试模型        |AP@IoU=0.5:0.95|AP@IoU=0.5|
| ------------ | ------------------ | ----------------------- | ------------- | -------- |
| SE5-16       | ppyolov3_opencv.py | ppyolov3_fp32_1b.bmodel        |    0.290 |    0.560 |
| SE5-16       | ppyolov3_opencv.py | ppyolov3_int8_1b.bmodel        |    0.267 |    0.536 |
| SE5-16       | ppyolov3_opencv.py | ppyolov3_fp32_1b.bmodel        |    0.290 |    0.560 |
| SE5-16       | ppyolov3_bmcv.py   | ppyolov3_int8_1b.bmodel        |    0.267 |    0.538 |
| SE5-16       | ppyolov3_bmcv.soc  | ppyolov3_fp32_1b.bmodel        |    0.278 |    0.546 |
| SE5-16       | ppyolov3_bmcv.soc  | ppyolov3_int8_1b.bmodel        |    0.255 |    0.525 |
| SE5-16       | ppyolov3_sail.soc  | ppyolov3_fp32_1b.bmodel        |    0.282 |    0.554 |
| SE5-16       | ppyolov3_sail.soc  | ppyolov3_int8_1b.bmodel        |    0.258 |    0.525 |
| SE7-32       | ppyolov3_opencv.py | ppyolov3_fp32_1b.bmodel        |    0.290 |    0.560 |
| SE7-32       | ppyolov3_opencv.py | ppyolov3_fp16_1b.bmodel        |    0.290 |    0.560 |
| SE7-32       | ppyolov3_opencv.py | ppyolov3_int8_1b.bmodel        |    0.285 |    0.556 |
| SE7-32       | ppyolov3_bmcv.py   | ppyolov3_fp32_1b.bmodel        |    0.289 |    0.559 |
| SE7-32       | ppyolov3_bmcv.py   | ppyolov3_fp16_1b.bmodel        |    0.289 |    0.559 |
| SE7-32       | ppyolov3_bmcv.py   | ppyolov3_int8_1b.bmodel        |    0.282 |    0.551 |
| SE7-32       | ppyolov3_bmcv.soc  | ppyolov3_fp32_1b.bmodel        |    0.279 |    0.548 |
| SE7-32       | ppyolov3_bmcv.soc  | ppyolov3_fp16_1b.bmodel        |    0.278 |    0.547 |
| SE7-32       | ppyolov3_bmcv.soc  | ppyolov3_int8_1b.bmodel        |    0.273 |    0.544 |
| SE7-32       | ppyolov3_sail.soc  | ppyolov3_fp32_1b.bmodel        |    0.281 |    0.551 |
| SE7-32       | ppyolov3_sail.soc  | ppyolov3_fp16_1b.bmodel        |    0.281 |    0.551 |
| SE7-32       | ppyolov3_sail.soc  | ppyolov3_int8_1b.bmodel        |    0.274 |    0.542 |
| SE9-16       | ppyolov3_opencv.py | ppyolov3_fp32_1b.bmodel        |    0.290 |    0.560 |
| SE9-16       | ppyolov3_opencv.py | ppyolov3_fp16_1b.bmodel        |    0.290 |    0.561 |
| SE9-16       | ppyolov3_opencv.py | ppyolov3_int8_1b.bmodel        |    0.285 |    0.556 |
| SE9-16       | ppyolov3_bmcv.py   | ppyolov3_fp32_1b.bmodel        |    0.289 |    0.559 |
| SE9-16       | ppyolov3_bmcv.py   | ppyolov3_fp16_1b.bmodel        |    0.289 |    0.559 |
| SE9-16       | ppyolov3_bmcv.py   | ppyolov3_int8_1b.bmodel        |    0.283 |    0.553 |
| SE9-16       | ppyolov3_bmcv.soc  | ppyolov3_fp32_1b.bmodel        |    0.278 |    0.546 |
| SE9-16       | ppyolov3_bmcv.soc  | ppyolov3_fp16_1b.bmodel        |    0.278 |    0.546 |
| SE9-16       | ppyolov3_bmcv.soc  | ppyolov3_int8_1b.bmodel        |    0.275 |    0.544 |
| SE9-16       | ppyolov3_sail.soc  | ppyolov3_fp32_1b.bmodel        |    0.281 |    0.552 |
| SE9-16       | ppyolov3_sail.soc  | ppyolov3_fp16_1b.bmodel        |    0.281 |    0.552 |
| SE9-16       | ppyolov3_sail.soc  | ppyolov3_int8_1b.bmodel        |    0.276 |    0.543 |
| SE9-16       | ppyolov3_opencv.py | ppyolov3_fp32_1b_2core.bmodel  |    0.290 |    0.560 |
| SE9-16       | ppyolov3_opencv.py | ppyolov3_fp16_1b_2core.bmodel  |    0.290 |    0.561 |
| SE9-16       | ppyolov3_opencv.py | ppyolov3_int8_1b_2core.bmodel  |    0.285 |    0.556 |
| SE9-16       | ppyolov3_bmcv.py   | ppyolov3_fp32_1b_2core.bmodel  |    0.289 |    0.559 |
| SE9-16       | ppyolov3_bmcv.py   | ppyolov3_fp16_1b_2core.bmodel  |    0.289 |    0.559 |
| SE9-16       | ppyolov3_bmcv.py   | ppyolov3_int8_1b_2core.bmodel  |    0.283 |    0.553 |
| SE9-16       | ppyolov3_bmcv.soc  | ppyolov3_fp32_1b_2core.bmodel  |    0.278 |    0.546 |
| SE9-16       | ppyolov3_bmcv.soc  | ppyolov3_fp16_1b_2core.bmodel  |    0.278 |    0.546 |
| SE9-16       | ppyolov3_bmcv.soc  | ppyolov3_int8_1b_2core.bmodel  |    0.275 |    0.544 |
| SE9-16       | ppyolov3_sail.soc  | ppyolov3_fp32_1b_2core.bmodel  |    0.281 |    0.552 |
| SE9-16       | ppyolov3_sail.soc  | ppyolov3_fp16_1b_2core.bmodel  |    0.281 |    0.552 |
| SE9-16       | ppyolov3_sail.soc  | ppyolov3_int8_1b_2core.bmodel  |    0.276 |    0.543 |
| SE9-8        | ppyolov3_opencv.py | ppyolov3_fp32_1b.bmodel        |    0.290 |    0.560 |
| SE9-8        | ppyolov3_opencv.py | ppyolov3_fp16_1b.bmodel        |    0.290 |    0.561 |
| SE9-8        | ppyolov3_opencv.py | ppyolov3_int8_1b.bmodel        |    0.285 |    0.556 |
| SE9-8        | ppyolov3_bmcv.py   | ppyolov3_fp32_1b.bmodel        |    0.289 |    0.559 |
| SE9-8        | ppyolov3_bmcv.py   | ppyolov3_fp16_1b.bmodel        |    0.289 |    0.559 |
| SE9-8        | ppyolov3_bmcv.py   | ppyolov3_int8_1b.bmodel        |    0.283 |    0.553 |
| SE9-8        | ppyolov3_bmcv.soc  | ppyolov3_fp32_1b.bmodel        |    0.278 |    0.546 |
| SE9-8        | ppyolov3_bmcv.soc  | ppyolov3_fp16_1b.bmodel        |    0.278 |    0.546 |
| SE9-8        | ppyolov3_bmcv.soc  | ppyolov3_int8_1b.bmodel        |    0.275 |    0.544 |
| SE9-8        | ppyolov3_sail.soc  | ppyolov3_fp32_1b.bmodel        |    0.281 |    0.552 |
| SE9-8        | ppyolov3_sail.soc  | ppyolov3_fp16_1b.bmodel        |    0.281 |    0.552 |
| SE9-8        | ppyolov3_sail.soc  | ppyolov3_int8_1b.bmodel        |    0.276 |    0.543 |

> **测试说明**：  
> 1. 由于sdk版本之间可能存在差异，实际运行结果与本表有<0.01的精度误差是正常的；
> 2. AP@IoU=0.5:0.95为area=all对应的指标；
> 3. 在搭载了相同TPU和SOPHONSDK的PCIe或SoC平台上，相同程序的精度一致，SE5系列对应BM1684，SE7系列对应BM1684X，SE9系列中，SE9-16对应BM1688，SE9-8对应CV186X；

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
| BM1684/ppyolov3_fp32_1b.bmodel     |          78.24  |
| BM1684/ppyolov3_int8_1b.bmodel     |          51.70  |
| BM1684X/ppyolov3_fp32_1b.bmodel    |         149.02  |
| BM1684X/ppyolov3_fp16_1b.bmodel    |          14.38  |
| BM1684X/ppyolov3_int8_1b.bmodel    |           7.24  |
| BM1688/ppyolov3_fp32_1b.bmodel     |         735.06  |
| BM1688/ppyolov3_fp16_1b.bmodel     |          90.44  |
| BM1688/ppyolov3_int8_1b.bmodel     |          27.32  |
| BM1688/ppyolov3_fp32_1b_2core.bmodel|         383.19  |
| BM1688/ppyolov3_fp16_1b_2core.bmodel|          53.11  |
| BM1688/ppyolov3_int8_1b_2core.bmodel|          20.16  |
| CV186X/ppyolov3_fp32_1b.bmodel     |         744.30  |
| CV186X/ppyolov3_fp16_1b.bmodel     |          97.62  |
| CV186X/ppyolov3_int8_1b.bmodel     |          32.15  |

> **测试说明**：  
1. 性能测试结果具有一定的波动性；
2. `calculate time`已折算为平均每张图片的推理时间；
3. SoC和PCIe的测试结果基本一致。

### 7.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++和Python例程打印的时间已经折算为单张图片的处理时间。

在不同的测试平台上，使用不同的例程、模型测试`datasets/coco/val2017_1000`，conf_thresh=0.5，nms_thresh=0.5，ppyolov3性能测试结果如下：
|    测试平台 |      测试程序       |        测试模型         |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ------------------ | ----------------------- | --------- | ---------- | ----------- | ----------- |
|   SE5-16    |ppyolov3_opencv.py |   ppyolov3_fp32_1b.bmodel    |      15.24      |      27.63      |      86.30      |      97.82      |
|   SE5-16    |ppyolov3_opencv.py |   ppyolov3_int8_1b.bmodel    |      15.21      |      27.72      |      59.65      |      97.15      |
|   SE5-16    | ppyolov3_bmcv.py  |   ppyolov3_fp32_1b.bmodel    |      3.64       |      2.32       |      83.97      |     110.51      |
|   SE5-16    | ppyolov3_bmcv.py  |   ppyolov3_int8_1b.bmodel    |      3.63       |      2.31       |      57.35      |     110.27      |
|   SE5-16    | ppyolov3_bmcv.soc |   ppyolov3_fp32_1b.bmodel    |      4.86       |      1.56       |      78.13      |      16.70      |
|   SE5-16    | ppyolov3_bmcv.soc |   ppyolov3_int8_1b.bmodel    |      4.88       |      1.57       |      51.60      |      16.80      |
|   SE5-16    | ppyolov3_sail.soc |   ppyolov3_fp32_1b.bmodel    |      3.27       |      3.08       |      79.04      |      15.78      |
|   SE5-16    | ppyolov3_sail.soc |   ppyolov3_int8_1b.bmodel    |      3.27       |      3.08       |      52.49      |      15.76      |
|   SE7-32    |ppyolov3_opencv.py |   ppyolov3_fp32_1b.bmodel    |      15.07      |      28.61      |     158.26      |      95.17      |
|   SE7-32    |ppyolov3_opencv.py |   ppyolov3_fp16_1b.bmodel    |      15.25      |      29.19      |      23.83      |      96.63      |
|   SE7-32    |ppyolov3_opencv.py |   ppyolov3_int8_1b.bmodel    |      15.33      |      28.58      |      16.63      |      96.36      |
|   SE7-32    | ppyolov3_bmcv.py  |   ppyolov3_fp32_1b.bmodel    |      3.16       |      1.79       |     155.52      |     107.93      |
|   SE7-32    | ppyolov3_bmcv.py  |   ppyolov3_fp16_1b.bmodel    |      3.15       |      1.79       |      20.91      |     108.05      |
|   SE7-32    | ppyolov3_bmcv.py  |   ppyolov3_int8_1b.bmodel    |      3.13       |      1.79       |      13.79      |     106.55      |
|   SE7-32    | ppyolov3_bmcv.soc |   ppyolov3_fp32_1b.bmodel    |      4.34       |      0.66       |     148.99      |      16.73      |
|   SE7-32    | ppyolov3_bmcv.soc |   ppyolov3_fp16_1b.bmodel    |      4.34       |      0.65       |      14.33      |      16.73      |
|   SE7-32    | ppyolov3_bmcv.soc |   ppyolov3_int8_1b.bmodel    |      4.34       |      0.66       |      7.19       |      16.74      |
|   SE7-32    | ppyolov3_sail.soc |   ppyolov3_fp32_1b.bmodel    |      2.76       |      2.57       |     149.88      |      15.79      |
|   SE7-32    | ppyolov3_sail.soc |   ppyolov3_fp16_1b.bmodel    |      2.76       |      2.56       |      15.21      |      15.80      |
|   SE7-32    | ppyolov3_sail.soc |   ppyolov3_int8_1b.bmodel    |      2.76       |      2.56       |      8.08       |      15.77      |
|   SE9-16    |ppyolov3_opencv.py |   ppyolov3_fp32_1b.bmodel    |      19.61      |      36.50      |     746.66      |     132.59      |
|   SE9-16    |ppyolov3_opencv.py |   ppyolov3_fp16_1b.bmodel    |      19.61      |      37.35      |     102.32      |     133.42      |
|   SE9-16    |ppyolov3_opencv.py |   ppyolov3_int8_1b.bmodel    |      19.60      |      37.42      |      39.15      |     133.85      |
|   SE9-16    | ppyolov3_bmcv.py  |   ppyolov3_fp32_1b.bmodel    |      4.45       |      3.95       |     743.54      |     150.03      |
|   SE9-16    | ppyolov3_bmcv.py  |   ppyolov3_fp16_1b.bmodel    |      4.44       |      3.95       |      98.82      |     150.03      |
|   SE9-16    | ppyolov3_bmcv.py  |   ppyolov3_int8_1b.bmodel    |      4.44       |      3.95       |      35.57      |     148.29      |
|   SE9-16    | ppyolov3_bmcv.soc |   ppyolov3_fp32_1b.bmodel    |      5.82       |      1.74       |     734.97      |      23.37      |
|   SE9-16    | ppyolov3_bmcv.soc |   ppyolov3_fp16_1b.bmodel    |      5.80       |      1.73       |      90.38      |      23.31      |
|   SE9-16    | ppyolov3_bmcv.soc |   ppyolov3_int8_1b.bmodel    |      5.79       |      1.73       |      27.23      |      23.29      |
|   SE9-16    | ppyolov3_sail.soc |   ppyolov3_fp32_1b.bmodel    |      3.92       |      5.16       |     737.36      |      22.00      |
|   SE9-16    | ppyolov3_sail.soc |   ppyolov3_fp16_1b.bmodel    |      3.92       |      5.17       |      92.64      |      21.99      |
|   SE9-16    | ppyolov3_sail.soc |   ppyolov3_int8_1b.bmodel    |      3.88       |      5.14       |      29.47      |      21.90      |
|   SE9-16    |ppyolov3_opencv.py |ppyolov3_fp32_1b_2core.bmodel |      19.58      |      38.72      |     395.18      |     133.81      |
|   SE9-16    |ppyolov3_opencv.py |ppyolov3_fp16_1b_2core.bmodel |      19.58      |      37.38      |      64.96      |     134.18      |
|   SE9-16    |ppyolov3_opencv.py |ppyolov3_int8_1b_2core.bmodel |      19.60      |      38.01      |      32.05      |     133.43      |
|   SE9-16    | ppyolov3_bmcv.py  |ppyolov3_fp32_1b_2core.bmodel |      4.47       |      3.94       |     391.58      |     149.92      |
|   SE9-16    | ppyolov3_bmcv.py  |ppyolov3_fp16_1b_2core.bmodel |      4.43       |      3.95       |      61.30      |     150.10      |
|   SE9-16    | ppyolov3_bmcv.py  |ppyolov3_int8_1b_2core.bmodel |      4.46       |      3.95       |      28.60      |     148.18      |
|   SE9-16    | ppyolov3_bmcv.soc |ppyolov3_fp32_1b_2core.bmodel |      5.85       |      1.72       |     383.13      |      23.34      |
|   SE9-16    | ppyolov3_bmcv.soc |ppyolov3_fp16_1b_2core.bmodel |      5.84       |      1.73       |      53.06      |      23.27      |
|   SE9-16    | ppyolov3_bmcv.soc |ppyolov3_int8_1b_2core.bmodel |      5.81       |      1.73       |      20.09      |      23.27      |
|   SE9-16    | ppyolov3_sail.soc |ppyolov3_fp32_1b_2core.bmodel |      3.90       |      5.17       |     385.47      |      22.03      |
|   SE9-16    | ppyolov3_sail.soc |ppyolov3_fp16_1b_2core.bmodel |      3.92       |      5.15       |      55.32      |      22.01      |
|   SE9-16    | ppyolov3_sail.soc |ppyolov3_int8_1b_2core.bmodel |      3.90       |      5.14       |      22.34      |      21.95      |
|    SE9-8    |ppyolov3_opencv.py |   ppyolov3_fp32_1b.bmodel    |      32.64      |      38.33      |     756.09      |     133.57      |
|    SE9-8    |ppyolov3_opencv.py |   ppyolov3_fp16_1b.bmodel    |      25.44      |      38.47      |     109.32      |     133.84      |
|    SE9-8    |ppyolov3_opencv.py |   ppyolov3_int8_1b.bmodel    |      19.68      |      38.21      |      43.81      |     133.42      |
|    SE9-8    | ppyolov3_bmcv.py  |   ppyolov3_fp32_1b.bmodel    |      4.31       |      3.81       |     752.84      |     149.62      |
|    SE9-8    | ppyolov3_bmcv.py  |   ppyolov3_fp16_1b.bmodel    |      4.30       |      3.83       |     105.92      |     149.82      |
|    SE9-8    | ppyolov3_bmcv.py  |   ppyolov3_int8_1b.bmodel    |      4.28       |      3.82       |      40.29      |     147.92      |
|    SE9-8    | ppyolov3_bmcv.soc |   ppyolov3_fp32_1b.bmodel    |      5.80       |      1.71       |     744.18      |      23.43      |
|    SE9-8    | ppyolov3_bmcv.soc |   ppyolov3_fp16_1b.bmodel    |      5.67       |      1.72       |      97.52      |      23.33      |
|    SE9-8    | ppyolov3_bmcv.soc |   ppyolov3_int8_1b.bmodel    |      5.71       |      1.71       |      32.03      |      23.32      |
|    SE9-8    | ppyolov3_sail.soc |   ppyolov3_fp32_1b.bmodel    |      4.67       |      5.07       |     746.61      |      22.05      |
|    SE9-8    | ppyolov3_sail.soc |   ppyolov3_fp16_1b.bmodel    |      5.93       |      5.07       |      99.83      |      22.02      |
|    SE9-8    | ppyolov3_sail.soc |   ppyolov3_int8_1b.bmodel    |      3.80       |      5.05       |      34.31      |      21.97      |

> **测试说明**：  
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. SE5-16/SE7-32的主控处理器均为8核CA53@2.3GHz，SE9-16为8核CA53@1.6GHz，SE9-8为6核CA53@1.6GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异，不同的阈值对后处理时间影响较大。 

## 8. FAQ
请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。
