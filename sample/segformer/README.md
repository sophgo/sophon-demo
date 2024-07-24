[简体中文](./README.md) | [English](./README_EN.md)

# SegFormer

* [1. 简介](#1-简介)
* [2. 特性](#2-特性)
* [3. 准备模型与数据](#3-准备模型与数据)
* [4. 模型编译](#4-模型编译)
* [5. 例程测试](#5-例程测试)

## 1. 简介
SegFormer是一种用于语义分割的简单、高效和强大的方法。SegFormer使用了Transformer技术，Transformer是一种用于序列建模的深度学习模型，它在自然语言处理中广泛应用。本例程对[​SegFormer官方开源仓库](https://github.com/NVlabs/SegFormer)版本的模型和算法进行移植，使之能在SOPHON BM1684\BM1684X\BM1688\CV186X上进行推理测试。

## 2. 特性
* 支持BM1688/CV186X(SoC)、BM1684X(x86 PCIe、SoC)和BM1684(x86 PCIe、SoC、arm PCIe)
* 支持FP32、FP16(BM1688/BM1684X/CV186X)模型编译和推理
* 支持基于BMCV预处理的C++推理
* 支持基于OpenCV和BMCV预处理的Python推理
* 支持图片和视频测试

## 3. 准备模型与数据
建议使用TPU-MLIR编译BModel，目前官方的Segfomer只有pth预训练模型，pth模型在编译前要导出成onnx模型。具体可参考[Segformer模型导出](./docs/Segformer_Export_Guide.md)。

同时，您需要准备用于测试的数据集，如果量化模型，还要准备用于量化的数据集。

本例程使用[Cityscapes](https://www.cityscapes-dataset.com/downloads/)进行测试，更多的公开数据集，请参考官方推荐[Prepare datasets](https://github.com/NVlabs/SegFormer/blob/master/docs/dataset_prepare.md)

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
│   └── segformer.b0.512x1024.city.160k_fp32_1b.bmodel               # 使用TPU-MLIR编译，用于BM1684的FP32 BModel， batch_size=1
├── BM1684X
│   ├── segformer.b0.512x1024.city.160k_fp32_1b.bmodel               # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=1
│   └── segformer.b0.512x1024.city.160k_fp16_1b.bmodel               # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=1
├── BM1688
│   ├── segformer.b0.512x1024.city.160k_fp32_1b_2core.bmodel         # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1, num_core=2
│   ├── segformer.b0.512x1024.city.160k_fp32_1b.bmodel               # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1, num_core=1
│   ├── segformer.b0.512x1024.city.160k_fp16_1b_2core.bmodel         # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1, num_core=2
│   └── segformer.b0.512x1024.city.160k_fp16_1b.bmodel               # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1, num_core=1
├── CV186X
│   ├── segformer.b0.512x1024.city.160k_fp32_1b.bmodel               # 使用TPU-MLIR编译，用于CV186X的FP32 BModel，batch_size=1
│   ├── segformer.b0.512x1024.city.160k_fp16_1b.bmodel               # 使用TPU-MLIR编译，用于CV186X的FP16 BModel，batch_size=1
└── onnx
    └── segformer.b0.512x1024.city.160k.onnx                         # pt导出的onnx动态模型
```
下载的数据包括：
```
./datasets
├── cali                                #量化图片
│   ├── xxx.png                                                                                 
├── cityscapes                          #测试图片集
│   ├── gtFine                          #评价图片 
│   ├── leftImg8bit                     #测试图片                
│   └── val.txt                         #评价图片列表
├── cityscapes_small                    #测试图片集—小
│   ├── gtFine                          #评价图片
│   └── leftImg8bit                     #测试图片
└── cityscapes_video.avi                #测试视频
```
## 4. 模型编译
导出的模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。建议使用TPU-MLIR编译BModel。

模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#2-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的SDK中获取)。

- 生成FP32 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684/BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684 #bm1684x/bm1688/cv186x

```

​执行上述命令会在`models/BM1684`下生成`segformer.b0.512x1024.city.160k_fp32_1b.bmodel`等文件，即转换好的FP32 BModel。


- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X**），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x #bm1688/cv186x
```

​执行上述命令会在`models/BM1684X/`下生成`segformer.b0.512x1024.city.160k_fp16_1b.bmodel`等文件，即转换好的FP16 BModel。



## 5. 推理测试
* [C++例程](cpp/README.md)
* [Python例程](python/README.md)


## 6. 精度测试
### 6.1 测试方法
首先，参考[C++例程](cpp/README.md#32-测试图片)或[Python例程](python/README.md#22-测试图片)推理要测试的数据集，生成预测的json文件，注意修改数据集(datasets/cityscapes)。
然后，使用`tools`目录下的`segformer_eval.py`脚本，将测试生成的json文件与测试集标签json文件进行对比，计算出目标检测的评价指标，命令如下：

```bash
# 请根据实际情况修改程序路径和json文件路径
python3 tools/segformer_eval.py --result_json python/results/segformer.b0.512x1024.city.160k_fp32_1b.bmodel_cityscapes_opencv_python_result.json
python3 tools/segformer_eval.py --result_json cpp/segformer_bmcv/results/segformer.b0.512x1024.city.160k_fp32_1b.bmodel_cityscapes_sail_cpp_result.json
```

### 6.2 测试结果
在cityscapes数据集上，其精度如下精度测试结果如下：
|   测试平台    |      测试程序       |                     测试模型                    |   mIoU   |   mAcc   |   aAcc   |
| ------------ | ------------------- | ---------------------------------------------- | -------- | -------- |----------|
| SE5-16       | segformer_opencv.py | segformer.b0.512x1024.city.160k_fp32_1b.bmodel |    68.35 |    76.96 |    94.75 |
| SE5-16       | segformer_bmcv.py   | segformer.b0.512x1024.city.160k_fp32_1b.bmodel |    68.05 |    76.60 |    94.68 |
| SE5-16       | segformer_bmcv.cpp  | segformer.b0.512x1024.city.160k_fp32_1b.bmodel |    68.33 |    76.94 |    94.75 |
| SE5-16       | segformer_sail.cpp  | segformer.b0.512x1024.city.160k_fp32_1b.bmodel |    68.24 |    76.82 |    94.73 |
| SE7-32       | segformer_opencv.py | segformer.b0.512x1024.city.160k_fp32_1b.bmodel |    68.35 |    76.96 |    94.75 |
| SE7-32       | segformer_bmcv.py   | segformer.b0.512x1024.city.160k_fp32_1b.bmodel |    68.04 |    76.58 |    94.68 |
| SE7-32       | segformer_bmcv.pcie | segformer.b0.512x1024.city.160k_fp32_1b.bmodel |    68.35 |    76.96 |    94.75 |
| SE7-32       | segformer_sail.pcie | segformer.b0.512x1024.city.160k_fp32_1b.bmodel |    68.35 |    76.96 |    94.75 |
| SE7-32       | segformer_opencv.py | segformer.b0.512x1024.city.160k_fp16_1b.bmodel |    68.34 |    76.95 |    94.75 |
| SE7-32       | segformer_bmcv.py   | segformer.b0.512x1024.city.160k_fp16_1b.bmodel |    68.02 |    76.53 |    94.68 |
| SE7-32       | segformer_bmcv.pcie | segformer.b0.512x1024.city.160k_fp16_1b.bmodel |    68.35 |    76.96 |    94.75 |
| SE7-32       | segformer_sail.pcie | segformer.b0.512x1024.city.160k_fp16_1b.bmodel |    68.34 |    76.96 |    94.75 |
| SE9-16       | segformer_opencv.py | segformer.b0.512x1024.city.160k_fp32_1b.bmodel |    68.35 |    76.96 |    94.75 |
| SE9-16       | segformer_opencv.py | segformer.b0.512x1024.city.160k_fp16_1b.bmodel |    68.34 |    76.95 |    94.75 |
| SE9-16       | segformer_bmcv.py   | segformer.b0.512x1024.city.160k_fp32_1b.bmodel |    68.17 |    76.70 |    94.70 |
| SE9-16       | segformer_bmcv.py   | segformer.b0.512x1024.city.160k_fp16_1b.bmodel |    68.17 |    76.70 |    94.70 |
| SE9-16       | segformer_bmcv.soc  | segformer.b0.512x1024.city.160k_fp32_1b.bmodel |    68.33 |    76.94 |    94.75 |
| SE9-16       | segformer_bmcv.soc  | segformer.b0.512x1024.city.160k_fp16_1b.bmodel |    68.32 |    76.93 |    94.75 |
| SE9-16       | segformer_sail.soc  | segformer.b0.512x1024.city.160k_fp32_1b.bmodel |    68.33 |    76.94 |    94.75 |
| SE9-16       | segformer_sail.soc  | segformer.b0.512x1024.city.160k_fp16_1b.bmodel |    68.32 |    76.93 |    94.75 |
| SE9-16       | segformer_opencv.py | segformer.b0.512x1024.city.160k_fp32_1b_2core.bmodel |    68.35 |    76.96 |    94.75 |
| SE9-16       | segformer_opencv.py | segformer.b0.512x1024.city.160k_fp16_1b_2core.bmodel |    68.34 |    76.95 |    94.75 |
| SE9-16       | segformer_bmcv.py   | segformer.b0.512x1024.city.160k_fp32_1b_2core.bmodel |    68.17 |    76.70 |    94.70 |
| SE9-16       | segformer_bmcv.py   | segformer.b0.512x1024.city.160k_fp16_1b_2core.bmodel |    68.17 |    76.70 |    94.70 |
| SE9-16       | segformer_bmcv.soc  | segformer.b0.512x1024.city.160k_fp32_1b_2core.bmodel |    68.33 |    76.94 |    94.75 |
| SE9-16       | segformer_bmcv.soc  | segformer.b0.512x1024.city.160k_fp16_1b_2core.bmodel |    68.32 |    76.93 |    94.75 |
| SE9-16       | segformer_sail.soc  | segformer.b0.512x1024.city.160k_fp32_1b_2core.bmodel |    68.33 |    76.94 |    94.75 |
| SE9-16       | segformer_sail.soc  | segformer.b0.512x1024.city.160k_fp16_1b_2core.bmodel |    68.32 |    76.93 |    94.75 |
| SE9-8        | segformer_opencv.py | segformer.b0.512x1024.city.160k_fp32_1b.bmodel |    68.35 |    76.96 |    94.75 |
| SE9-8        | segformer_opencv.py | segformer.b0.512x1024.city.160k_fp16_1b.bmodel |    68.34 |    76.95 |    94.75 |
| SE9-8        | segformer_bmcv.py   | segformer.b0.512x1024.city.160k_fp32_1b.bmodel |    68.17 |    76.70 |    94.70 |
| SE9-8        | segformer_bmcv.py   | segformer.b0.512x1024.city.160k_fp16_1b.bmodel |    68.17 |    76.70 |    94.70 |
| SE9-8        | segformer_sail.soc  | segformer.b0.512x1024.city.160k_fp32_1b.bmodel |    68.33 |    76.94 |    94.75 |
| SE9-8        | segformer_sail.soc  | segformer.b0.512x1024.city.160k_fp16_1b.bmodel |    68.32 |    76.93 |    94.75 |

> **测试说明**：  
> 1. batch_size=4和batch_size=1的模型精度一致；
> 2. SoC和PCIe的模型精度一致；

## 7. 性能测试
### 7.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684/segformer.b0.512x1024.city.160k_fp32_1b.bmodel
```
在cityscapes测试各个模型的理论推理时间，结果如下：
|                          测试模型                           | calculate time(ms) |
| ---------------------------------------------------------- | -------------------|
| SE5-16 /segformer.b0.512x1024.city.160k_fp32_1b.bmodel     | 365.63             |
| SE7-32 /segformer.b0.512x1024.city.160k_fp32_1b.bmodel     | 288.866            |
| SE7-32 /segformer.b0.512x1024.city.160k_fp16_1b.bmodel     | 54.229             |
| SE9-16/segformer.b0.512x1024.city.160k_fp32_1b.bmodel      | 413.63             |
| SE9-16/segformer.b0.512x1024.city.160k_fp16_1b.bmodel      | 119.02             |
| SE9-16/segformer.b0.512x1024.city.160k_fp32_1b_2core.bmodel| 288.31             |
| SE9-16/segformer.b0.512x1024.city.160k_fp16_1b_2core.bmodel| 104.02             |
| SE9-8/segformer.b0.512x1024.city.160k_fp32_1b.bmodel       | 473.95             |
| SE9-8/segformer.b0.512x1024.city.160k_fp16_1b.bmodel       | 157.45             |

> **测试说明**：  
> 1. 性能测试结果具有一定的波动性；
> 2. `calculate time`已折算为平均每张图片的推理时间；
> 3. SoC和PCIe的测试结果基本一致。


### 7.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++例程打印的预处理时间、推理时间、后处理时间为整个batch处理的时间，需除以相应的batch size才是每张图片的处理时间。

在不同的测试平台上，使用不同的例程、模型测试`datasets/cityscapes`,性能测试结果如下：

|  测试平台     |    测试程序         |                 测试模型                        |decode_time|preprocess_time|inference_time |postprocess_time| 
| -----------  | ------------------- | ---------------------------------------------- | --------- | ------------- | ------------- | -------------- |
|    SE5-16    | segformer_opencv.py | segformer.b0.512x1024.city.160k_fp32_1b.bmodel |   110.23  |      23.15    |     389.67    |     182.70     |
|    SE5-16    | segformer_bmcv.py   | segformer.b0.512x1024.city.160k_fp32_1b.bmodel |   140.67  |      5.93     |     369.88    |     141.78     |
|    SE5-16    | segformer_bmcv.soc  | segformer.b0.512x1024.city.160k_fp32_1b.bmodel |   126.51  |      1.52     |     365.50    |     261.71     |
|    SE5-16    | segformer_sail.soc  | segformer.b0.512x1024.city.160k_fp32_1b.bmodel |   106.44  |      7.14     |     365.86    |     259.91     |
|    SE7-32    |segformer_opencv.py  |segformer.b0.512x1024.city.160k_fp32_1b.bmodel  |   145.98  |      21.14    |     320.01    |     166.92     |
|    SE7-32    |segformer_opencv.py  |segformer.b0.512x1024.city.160k_fp16_1b.bmodel  |   153.89  |      21.21    |      74.10    |     167.25     |
|    SE7-32    | segformer_bmcv.py   |segformer.b0.512x1024.city.160k_fp32_1b.bmodel  |   154.53  |      5.19     |     303.69    |     123.74     |
|    SE7-32    | segformer_bmcv.py   |segformer.b0.512x1024.city.160k_fp16_1b.bmodel  |   146.50  |      5.15     |      57.59    |     123.58     |
|    SE7-32    |segformer_bmcv.soc   |segformer.b0.512x1024.city.160k_fp32_1b.bmodel  |   161.21  |      1.54     |     300.34    |     249.28     |
|    SE7-32    |segformer_bmcv.soc   |segformer.b0.512x1024.city.160k_fp16_1b.bmodel  |   148.56  |      1.54     |      54.27    |     249.03     |
|    SE7-32    |segformer_sail.soc   |segformer.b0.512x1024.city.160k_fp32_1b.bmodel  |   154.23  |      5.95     |     300.70    |     250.16     |
|    SE7-32    |segformer_sail.soc   |segformer.b0.512x1024.city.160k_fp16_1b.bmodel  |   152.21  |      5.96     |      54.60    |     249.54     |
|    SE9-16    |segformer_opencv.py  | segformer.b0.512x1024.city.160k_fp32_1b.bmodel |   156.92  |      27.17    |     447.52    |   248.12       |
|    SE9-16    |segformer_opencv.py  | segformer.b0.512x1024.city.160k_fp16_1b.bmodel |   162.62  |      27.15    |     153.11    |   250.05       |
|    SE9-16    | segformer_bmcv.py   | segformer.b0.512x1024.city.160k_fp32_1b.bmodel |   183.98  |      12.02    |     419.88    |   194.48       |
|    SE9-16    | segformer_bmcv.py   | segformer.b0.512x1024.city.160k_fp16_1b.bmodel |   185.24  |      12.03    |     125.12    |   194.61       |
|    SE9-16    |segformer_bmcv.soc   | segformer.b0.512x1024.city.160k_fp32_1b.bmodel |   207.84  |      4.28     |     413.60    |     344.03     |
|    SE9-16    |segformer_bmcv.soc   | segformer.b0.512x1024.city.160k_fp16_1b.bmodel |   207.85  |      4.28     |     118.95    |     348.56     |
|    SE9-16    |segformer_sail.soc   | segformer.b0.512x1024.city.160k_fp32_1b.bmodel |   179.58  |      13.25    |     414.41    |     338.35     |
|    SE9-16    |segformer_sail.soc   | segformer.b0.512x1024.city.160k_fp16_1b.bmodel |   181.23  |      13.22    |     119.71    |     338.43     |
|    SE9-16    |segformer_opencv.py  |segformer.b0.512x1024.city.160k_fp32_1b_2core.bmodel|   160.19    |    28.94    |   322.68    |   253.40     |
|    SE9-16    |segformer_opencv.py  |segformer.b0.512x1024.city.160k_fp16_1b_2core.bmodel|   163.17    |    27.50    |   139.36    |   252.66     |
|    SE9-16    | segformer_bmcv.py   |segformer.b0.512x1024.city.160k_fp32_1b_2core.bmodel|   183.64    |    12.06    |   294.23    |   195.08     |
|    SE9-16    | segformer_bmcv.py   |segformer.b0.512x1024.city.160k_fp16_1b_2core.bmodel|   184.64    |    12.06    |   110.32    |   194.97     |
|    SE9-16    |segformer_bmcv.soc   |segformer.b0.512x1024.city.160k_fp32_1b_2core.bmodel|   206.37    |    4.28     |   288.10    |   347.99     |
|    SE9-16    |segformer_bmcv.soc   |segformer.b0.512x1024.city.160k_fp16_1b_2core.bmodel|   206.85    |    4.28     |   104.27    |   345.63     |
|    SE9-16    |segformer_sail.soc   |segformer.b0.512x1024.city.160k_fp32_1b_2core.bmodel|   184.14    |    13.25    |   288.70    |   339.33     |
|    SE9-16    |segformer_sail.soc   |segformer.b0.512x1024.city.160k_fp16_1b_2core.bmodel|   186.15    |    13.27    |   105.01    |   338.84     |
|    SE9-8     |segformer_opencv.py  | segformer.b0.512x1024.city.160k_fp32_1b.bmodel |   157.53  |      29.47    |     508.93    |   223.66       |
|    SE9-8     |segformer_opencv.py  | segformer.b0.512x1024.city.160k_fp16_1b.bmodel |   164.70  |      29.17    |     190.92    |   224.96       |
|    SE9-8     | segformer_bmcv.py   | segformer.b0.512x1024.city.160k_fp32_1b.bmodel |   185.69  |      11.82    |     481.06    |   192.43       |
|    SE9-8     | segformer_bmcv.py   | segformer.b0.512x1024.city.160k_fp16_1b.bmodel |   185.56  |      11.89    |     162.81    |   197.55       |
|    SE9-8     |segformer_sail.soc   | segformer.b0.512x1024.city.160k_fp32_1b.bmodel |   183.28  |      12.80    |     413.78    |   341.59       |
|    SE9-8     |segformer_sail.soc   | segformer.b0.512x1024.city.160k_fp16_1b.bmodel |   183.28  |      12.80    |     413.78    |     341.59     |


> **测试说明**：  
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. SE5-16/SE7-32的主控处理器均为8核CA53@2.3GHz，SE9-16为8核CA53@1.6GHz，SE9-8为6核CA53@1.6GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异。 

## 8. FAQ
其他问题请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。
