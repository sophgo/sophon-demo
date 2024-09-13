# DirectMHP

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
​​DirectMHP 是一种新颖的单级端到端网络，专注于全范围的多人头部姿势估计，通过联合回归位置和方向来直接预测图像中所有人类头部的姿势。本例程对[​DirectMHP官方开源仓库](https://github.com/hnuzhy/DirectMHP)的模型和算法进行移植，使之能在SOPHON BM1684X/BM1688/CV186X上进行推理测试。

## 2. 特性
* 支持BM1688/CV186X(SoC)、BM1684X(x86 PCIe、SoC)
* 支持FP32、FP16(BM1684X/BM1688/CV186X)
* 支持基于BMCV预处理的C++推理
* 支持基于OpenCV和BMCV预处理的Python推理
* 支持单batch和多batch模型推理
* 支持1个输出模型推理
* 支持图片和视频测试

## 3. 准备模型与数据
建议使用TPU-MLIR编译BModel，在使用TPU-MLIR编译前需要导出ONNX模型。具体可参考[DirectMHP模型导出](./docs/DirectMHP_Export_Guide.md)。

​同时，您需要准备用于测试的数据集，如果量化模型，还要准备用于量化的数据集。

​本例程在`scripts`目录下提供了相关模型和数据的下载脚本`download.sh`，您也可以自己准备模型和数据集，并参考[4. 模型编译](#4-模型编译)进行模型转换。

```bash
# 安装unzip，若已安装请跳过，非ubuntu系统视情况使用yum或其他方式安装
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```

执行后，模型保存至`models/`，测试数据集下载并解压至`datasets/test/`，精度测试数据集下载并解压至`datasets/coco/`

```
下载的模型包括：
./models
├── BM1684X
│   ├── directmhp_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=1
│   └── directmhp_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=1
├── BM1688
│   ├── directmhp_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1, num_core=1
│   ├── directmhp_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1, num_core=1
│   ├── directmhp_fp32_1b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1, num_core=2
│   └── directmhp_fp16_1b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1, num_core=2
├── CV186X
│   ├── directmhp_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于CV186X的FP32 BModel，batch_size=1
│   └── directmhp_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于CV186X的FP16 BModel，batch_size=1
│── torch
│   └── directmhp_torchscript.pt   # trace后的torchscript模型
└── onnx
     └── dierctmhp.onnx           # 导出的动态onnx模型

    
         
```
下载的数据包括：
```
./datasets
├── test                                      # 测试图片
├── person_small.mp4                          # 测试视频
└── coco                                      
    ├── val                                   # coco val数据集：从CMU数据集中抽取的778张图片
    └── coco_style_sampled_val.json           # coco_style_sampled_val.json数据集标签文件，用于计算精度评价指标 
```

## 4. 模型编译
导出的模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。建议使用TPU-MLIR编译BModel。

模型编译前需要安装TPU-MLIR，本例程使用的TPU-MLIR版本是`v1.8`，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的SDK中获取)。

- 生成FP32 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684x #bm1688/cv186x
```

​执行上述命令会在`models/BM1684X`下生成`directmhp_fp32_1b.bmodel`文件，即转换好的FP32 BModel。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x #bm1688/cv186x
```

​执行上述命令会在`models/BM1684X/`下生成`directmhp_fp16_1b.bmodel`文件，即转换好的FP16 BModel。


## 5. 例程测试
- [C++例程](./cpp/README.md)
- [Python例程](./python/README.md)

## 6. 精度测试
### 6.1 测试方法

首先，参考[C++例程](cpp/README.md#32-测试图片)或[Python例程](python/README.md#22-测试图片)推理要测试的数据集，生成预测的json文件，注意修改数据集(datasets/coco/val)和相关参数(conf_thresh=0.001、nms_thresh=0.65)。  
然后，使用`tools`目录下的`eval.py`脚本，将测试生成的json文件与测试集标签json文件进行对比，计算出目标检测的评价指标，命令如下：
```bash
# 安装pycocotools，若已安装请跳过
pip3 install pycocotools
# 请根据实际情况修改程序路径和json文件路径
python3 tools/eval.py --gt_path datasets/coco/coco_style_sampled_val.json --result_json results/directmhp_fp32_1b.bmodel_val_opencv_python_result.json
```
### 6.2 测试结果
在val数据集上，精度测试结果如下：
|   测试平台    |         测试程序          |                   测试模型                |AP@IoU=0.5:0.95|    MAE    |
| ------------ | ------------------------- | ---------------------------------------- | ---------| -------- |
| SE7-32       | directmhp_opencv.py       | directmhp_fp32_1b.bmodel                 |    0.856 |    8.706 |
| SE7-32       | directmhp_opencv.py       | directmhp_fp16_1b.bmodel                 |    0.857 |    8.697 |
| SE7-32       | directmhp_bmcv.py         | directmhp_fp32_1b.bmodel                 |    0.856 |    8.758 |
| SE7-32       | directmhp_bmcv.py         | directmhp_fp16_1b.bmodel                 |    0.855 |    8.751 |
| SE7-32       | directmhp_bmcv.soc        | directmhp_fp32_1b.bmodel                 |    0.858 |    8.712 |
| SE7-32       | directmhp_bmcv.soc        | directmhp_fp16_1b.bmodel                 |    0.859 |    8.710 |
| SE9-16       | directmhp_opencv.py       | directmhp_fp32_1b.bmodel                 |    0.856 |    8.706 |
| SE9-16       | directmhp_opencv.py       | directmhp_fp16_1b.bmodel                 |    0.857 |    8.698 |
| SE9-16       | directmhp_bmcv.py         | directmhp_fp32_1b.bmodel                 |    0.857 |    8.730 |
| SE9-16       | directmhp_bmcv.py         | directmhp_fp16_1b.bmodel                 |    0.858 |    8.729 |
| SE9-16       | directmhp_bmcv.soc        | directmhp_fp32_1b.bmodel                 |    0.858 |    8.712 |
| SE9-16       | directmhp_bmcv.soc        | directmhp_fp16_1b.bmodel                 |    0.859 |    8.713 |
| SE9-16       | directmhp_opencv.py       | directmhp_fp32_1b_2core.bmodel           |    0.856 |    8.706 |
| SE9-16       | directmhp_opencv.py       | directmhp_fp16_1b_2core.bmodel           |    0.857 |    8.698 |
| SE9-16       | directmhp_bmcv.py         | directmhp_fp32_1b_2core.bmodel           |    0.857 |    8.730 |
| SE9-16       | directmhp_bmcv.py         | directmhp_fp16_1b_2core.bmodel           |    0.858 |    8.729 |
| SE9-16       | directmhp_bmcv.soc        | directmhp_fp32_1b_2core.bmodel           |    0.858 |    8.712 |
| SE9-16       | directmhp_bmcv.soc        | directmhp_fp16_1b_2core.bmodel           |    0.859 |    8.713 |
| SE9-8        | directmhp_opencv.py       | directmhp_fp32_1b.bmodel                 |    0.856 |    8.706 |
| SE9-8        | directmhp_opencv.py       | directmhp_fp16_1b.bmodel                 |    0.857 |    8.698 |
| SE9-8        | directmhp_bmcv.py         | directmhp_fp32_1b.bmodel                 |    0.857 |    8.730 |
| SE9-8        | directmhp_bmcv.py         | directmhp_fp16_1b.bmodel                 |    0.858 |    8.729 |
| SE9-8        | directmhp_bmcv.soc        | directmhp_fp32_1b.bmodel                 |    0.858 |    8.712 |
| SE9-8        | directmhp_bmcv.soc        | directmhp_fp16_1b.bmodel                 |    0.859 |    8.713 |


> **测试说明**：  
> 1. 由于sdk版本之间可能存在差异，实际运行结果与本表有<0.01的精度误差是正常的；
> 2. AP@IoU=0.5:0.95为area=all对应的指标。
> 3. 在搭载了相同TPU和SOPHONSDK的PCIe或SoC平台上，相同程序的精度一致，SE7系列对应BM1684X，SE9系列中，SE9-16对应BM1688，SE9-8对应CV186X；


## 7. 性能测试
### 7.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684X/directmhp_fp32_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

|              测试模型                | calculate time(ms) |
| ------------------------------------ | ----------------- |
| BM1684X/directmhp_fp32_1b.bmodel     |         84.34     |
| BM1684X/directmhp_fp16_1b.bmodel     |         23.48     |
| BM1688/directmhp_fp32_1b.bmodel      |         407.81    |
| BM1688/directmhp_fp16_1b.bmodel      |         107.39    |
| BM1688/directmhp_fp32_1b_2core.bmodel|         215.63    |
| BM1688/directmhp_fp16_1b_2core.bmodel|          62.98    |
| CV186X/directmhp_fp32_1b.bmodel      |         420.19    |
| CV186X/directmhp_fp16_1b.bmodel      |         115.67    |



> **测试说明**：  
1. 性能测试结果具有一定的波动性；
2. `calculate time`已折算为平均每张图片的推理时间；
3. SoC和PCIe的测试结果基本一致。


### 7.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++和Python例程打印的时间已经折算为单张图片的处理时间。

在不同的测试平台上，使用不同的例程、模型测试`datasets/coco/val`，conf_thresh=0.5，nms_thresh=0.5，性能测试结果如下：
|    测试平台  |     测试程序      |             测试模型              |    decode_time   | preprocess_time | inference_time | postprocess_time | 
| ----------- | ---------------- | ---------------------------------- | --------------  | ------------    | -----------     | ----------      |
|   SE7-32    |directmhp_opencv.py|     directmhp_fp32_1b.bmodel      |      35.46      |     103.07      |     102.92      |      5.95       |
|   SE7-32    |directmhp_opencv.py|     directmhp_fp16_1b.bmodel      |      40.34      |     101.64      |      41.95      |      5.96       |
|   SE7-32    | directmhp_bmcv.py |     directmhp_fp32_1b.bmodel      |      8.02       |      8.58       |      88.22      |      5.97       |
|   SE7-32    | directmhp_bmcv.py |     directmhp_fp16_1b.bmodel      |      6.56       |      8.68       |      27.41      |      5.97       |
|   SE7-32    |directmhp_bmcv.soc |     directmhp_fp32_1b.bmodel      |      9.95       |      4.06       |      84.32      |      2.74       |
|   SE7-32    |directmhp_bmcv.soc |     directmhp_fp16_1b.bmodel      |      9.90       |      4.06       |      23.43      |      2.74       |
|   SE9-16    |directmhp_opencv.py|     directmhp_fp32_1b.bmodel      |      54.05      |     134.03      |     430.60      |      7.30       |
|   SE9-16    |directmhp_opencv.py|     directmhp_fp16_1b.bmodel      |      51.51      |     137.22      |     131.09      |      7.31       |
|   SE9-16    | directmhp_bmcv.py |     directmhp_fp32_1b.bmodel      |      15.49      |      17.67      |     413.37      |      7.35       |
|   SE9-16    | directmhp_bmcv.py |     directmhp_fp16_1b.bmodel      |      11.31      |      17.65      |     112.64      |      7.35       |
|   SE9-16    |directmhp_bmcv.soc |     directmhp_fp32_1b.bmodel      |      15.30      |      7.15       |     407.72      |      3.32       |
|   SE9-16    |directmhp_bmcv.soc |     directmhp_fp16_1b.bmodel      |      15.17      |      7.15       |     107.29      |      3.28       |
|   SE9-16    |directmhp_opencv.py|  directmhp_fp32_1b_2core.bmodel   |      46.36      |     136.42      |     238.36      |      7.30       |
|   SE9-16    |directmhp_opencv.py|  directmhp_fp16_1b_2core.bmodel   |      49.97      |     134.85      |      86.90      |      7.30       |
|   SE9-16    | directmhp_bmcv.py |  directmhp_fp32_1b_2core.bmodel   |      12.88      |      17.63      |     221.12      |      7.37       |
|   SE9-16    | directmhp_bmcv.py |  directmhp_fp16_1b_2core.bmodel   |      11.33      |      17.63      |      68.06      |      7.36       |
|   SE9-16    |directmhp_bmcv.soc |  directmhp_fp32_1b_2core.bmodel   |      15.45      |      7.14       |     215.53      |      3.31       |
|   SE9-16    |directmhp_bmcv.soc |  directmhp_fp16_1b_2core.bmodel   |      15.13      |      7.15       |      62.87      |      3.27       |
|   SE9-8     |directmhp_opencv.py|     directmhp_fp32_1b.bmodel      |      50.38      |     139.18      |     442.47      |      7.37       |
|   SE9-8     |directmhp_opencv.py|     directmhp_fp16_1b.bmodel      |      58.76      |     140.77      |     137.92      |      7.41       |
|   SE9-8     | directmhp_bmcv.py |     directmhp_fp32_1b.bmodel      |      26.04      |      17.28      |     425.40      |      7.47       |
|   SE9-8     | directmhp_bmcv.py |     directmhp_fp16_1b.bmodel      |      24.99      |      17.27      |     120.67      |      7.49       |
|   SE9-8     |directmhp_bmcv.soc |     directmhp_fp32_1b.bmodel      |      17.18      |      7.16       |     420.06      |      3.45       |
|   SE9-8     |directmhp_bmcv.soc |     directmhp_fp16_1b.bmodel      |      15.77      |      7.16       |     115.50      |      3.42       |


> **测试说明**：  
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. SE7-32的主控处理器为8核CA53@2.3GHz，SE9-16为8核CA53@1.6GHz，SE9-8为6核CA53@1.6GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异，不同的阈值对后处理时间影响较大。 


## 8. FAQ
请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。