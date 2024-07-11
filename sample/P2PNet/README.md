# P2PNet

## 目录

- [P2PNet](#p2pnet)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
  - [3. 准备模型与数据](#3-准备模型与数据)
  - [4. 模型编译](#4-模型编译)
  - [5. 例程测试](#5-例程测试)
  - [6. 精度测试](#6-精度测试)
    - [6.1 测试方法](#61-测试方法)
    - [6.2 测试结果](#62-测试结果)
  - [7. 性能测试](#7-性能测试)
    - [7.1 bmrt\_test](#71-bmrt_test)
    - [7.2 程序运行性能](#72-程序运行性能)
  - [8. FAQ](#8-faq)

## 1. 简介
P2PNet是腾讯优图实验室提出的点对点网络（Point-to-Point Network，P2PNet），业界首创直接预测人头中心点的人群计数新范式，能够同时实现人群个体定位和人群计数，该算法在 2020 年 12 月份刷新 NWPU 榜单。本例程对[P2PNet官方开源仓库](https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet)的模型和算法进行移植，使之能在SOPHON BM1684|BM1684X|BM1688|CV186X上进行推理测试。

**数据集**: (https://www.datafountain.cn/datasets/5670)

## 2. 特性
* 支持BM1684X(x86 PCIe、SoC)、BM1684(x86 PCIe、SoC、arm PCIe)、BM1688(SoC)和CV186X(SoC)
* 支持FP32、FP16(BM1684X/BM1688/CV186X)、INT8模型编译和推理
* 支持基于BMCV预处理的C++推理
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
./models
├── BM1684
│   ├── p2pnet_bm1684_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684的FP32 BModel，batch_size=1
│   ├── p2pnet_bm1684_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684的INT8 BModel，batch_size=1
│   └── p2pnet_bm1684_int8_4b.bmodel   # 使用TPU-MLIR编译，用于BM1684的INT8 BModel，batch_size=4
├── BM1684X
│   ├── p2pnet_bm1684x_fp32_1b.bmodel  # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=1
│   ├── p2pnet_bm1684x_fp16_1b.bmodel  # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=1
│   ├── p2pnet_bm1684x_int8_1b.bmodel  # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=1
│   └── p2pnet_bm1684x_int8_4b.bmodel  # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=4
├── BM1688
│   ├── p2pnet_bm1688_fp32_1b.bmodel  # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1
│   ├── p2pnet_bm1688_fp16_1b.bmodel  # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1
│   ├── p2pnet_bm1688_int8_1b.bmodel  # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=1
│   └── p2pnet_bm1688_int8_4b.bmodel  # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=4
├── CV186X
│   ├── p2pnet_cv186x_fp32_1b.bmodel  # 使用TPU-MLIR编译，用于CV186X的FP32 BModel，batch_size=1
│   ├── p2pnet_cv186x_fp16_1b.bmodel  # 使用TPU-MLIR编译，用于CV186X的FP16 BModel，batch_size=1
│   ├── p2pnet_cv186x_int8_1b.bmodel  # 使用TPU-MLIR编译，用于CV186X的INT8 BModel，batch_size=1
│   └── p2pnet_cv186x_int8_4b.bmodel  # 使用TPU-MLIR编译，用于CV186X的INT8 BModel，batch_size=4
└── onnx
    ├── p2pnet_1b.onnx                 # onnx模型，batch_size=1
    └── p2pnet_4b.onnx                 # onnx模型，batch_size=4
```
下载的数据包括：
```
./datasets
├── test                      # 测试数据集
│   ├──ground-truth           # 用于计算评价指标
│   └──images                 # 测试图片
├── calibration               # 用于模型量化
└── video
    └──video.avi              # 测试视频
```

## 4. 模型编译
导出的模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。建议使用TPU-MLIR编译BModel。

模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#2-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index/material/all/all.html)相应版本的SDK中获取)。

- 生成FP32 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（支持BM1684|BM1684X|BM1688|CV186X），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684
#or
./scripts/gen_fp32bmodel_mlir.sh bm1684x
```

​执行上述命令会在`models/BM1684/`或`models/BM1684X`下生成`p2pnet_bm1684*_fp32_1b.bmodel`文件，即转换好的FP32 BModel。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（支持BM1684|BM1684X|BM1688|CV186X），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x
```

​执行上述命令会在`models/BM1684X/`下生成`p2pnet_bm1684x_fp16_1b.bmodel`文件，即转换好的FP16 BModel。

- 生成INT8 BModel

​本例程在`scripts`目录下提供了量化INT8 BModel的脚本，请注意修改`gen_int8bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台（支持BM1684|BM1684X|BM1688|CV186X），如：

```shell
./scripts/gen_int8bmodel_mlir.sh bm1684
#或
./scripts/gen_int8bmodel_mlir.sh bm1684x
```

​上述脚本会在`models/BM1684`或`models/BM1684X`下生成`p2pnet_bm1684*_int8_1b.bmodel`等文件，即转换好的INT8 BModel。

## 5. 例程测试
- [C++例程](./cpp/README.md)
- [Python例程](./python/README.md)

## 6. 精度测试
### 6.1 测试方法

首先，参考[C++例程](cpp/README.md#69-测试数据集)或[Python例程](python/README.md#47-测试数据集)推理要测试的数据集，生成预测的txt文件。
然后，使用`tools`目录下的`eval_acc.py`脚本，将测试生成的txt文件与测试集标签mat文件进行对比，计算出准确率信息，命令如下：
```bash
# 请根据实际情况修改ground-truth和测试结果路径
python3 tools/eval_acc.py --gt_path datasets/test/ground-truth --result_path python/results/images
```
### 6.2 测试结果
根据本例程提供的数据集，测试结果如下：
|    测试平台    |         测试程序       |      	     测试模型         	  |   MAE  |  MSE  |
| ------------ | --------------------- | ------------------------------ | ------ | ----- |
| SE5-16  | p2pnet_opencv.py      | p2pnet_bm1684_fp32_1b.bmodel   |  18.35 | 29.12 |
| SE5-16  | p2pnet_opencv.py      | p2pnet_bm1684_int8_1b.bmodel   |  20.44 | 32.36 |
| SE5-16  | p2pnet_opencv.py      | p2pnet_bm1684_int8_4b.bmodel   |  20.44 | 32.36 |
| SE5-16  | p2pnet_bmcv.py        | p2pnet_bm1684_fp32_1b.bmodel   |  20.20 | 30.47 |
| SE5-16  | p2pnet_bmcv.py        | p2pnet_bm1684_int8_1b.bmodel   |  20.66 | 32.96 |
| SE5-16  | p2pnet_bmcv.py        | p2pnet_bm1684_int8_4b.bmodel   |  20.66 | 32.96 |
| SE5-16  | p2pnet_bmcv.soc      | p2pnet_bm1684_fp32_1b.bmodel   |  18.21 | 28.70 |
| SE5-16  | p2pnet_bmcv.soc      | p2pnet_bm1684_int8_1b.bmodel   |  19.91 | 31.35 |
| SE5-16  | p2pnet_bmcv.soc      | p2pnet_bm1684_int8_4b.bmodel   |  19.91 | 31.35 |
| SE7-32 | p2pnet_opencv.py      | p2pnet_bm1684x_fp32_1b.bmodel  |  18.35 | 29.12 |
| SE7-32 | p2pnet_opencv.py      | p2pnet_bm1684x_fp16_1b.bmodel  |  18.34 | 29.11 |
| SE7-32 | p2pnet_opencv.py      | p2pnet_bm1684x_int8_1b.bmodel  |  18.49 | 29.64 |
| SE7-32 | p2pnet_opencv.py      | p2pnet_bm1684x_int8_4b.bmodel  |  18.49 | 29.64 |
| SE7-32 | p2pnet_bmcv.py        | p2pnet_bm1684x_fp32_1b.bmodel  |  20.21 | 30.49 |
| SE7-32 | p2pnet_bmcv.py        | p2pnet_bm1684x_fp16_1b.bmodel  |  20.21 | 30.49 |
| SE7-32 | p2pnet_bmcv.py        | p2pnet_bm1684x_int8_1b.bmodel  |  20.34 | 30.71 |
| SE7-32 | p2pnet_bmcv.py        | p2pnet_bm1684x_int8_4b.bmodel  |  20.34 | 30.71 |
| SE7-32 | p2pnet_bmcv.soc      | p2pnet_bm1684x_fp32_1b.bmodel  |  18.06 | 28.48 |
| SE7-32 | p2pnet_bmcv.soc      | p2pnet_bm1684x_fp16_1b.bmodel  |  18.15 | 28.59 |
| SE7-32 | p2pnet_bmcv.soc      | p2pnet_bm1684x_int8_1b.bmodel  |  18.01 | 28.31 |
| SE7-32 | p2pnet_bmcv.soc      | p2pnet_bm1684x_int8_4b.bmodel  |  17.99 | 28.32 |
| SE9-16 | p2pnet_opencv.py      | p2pnet_bm1688_fp32_1b.bmodel  |  18.35 | 29.12 |
| SE9-16 | p2pnet_opencv.py      | p2pnet_bm1688_fp16_1b.bmodel  |  18.33 | 29.11 |
| SE9-16 | p2pnet_opencv.py      | p2pnet_bm1688_int8_1b.bmodel  |  18.42 | 29.54 |
| SE9-16 | p2pnet_opencv.py      | p2pnet_bm1688_int8_4b.bmodel  |  18.42 | 29.54 |
| SE9-16 | p2pnet_bmcv.py        | p2pnet_bm1688_fp32_1b.bmodel  |  20.15 | 30.40 |
| SE9-16 | p2pnet_bmcv.py        | p2pnet_bm1688_fp16_1b.bmodel  |  20.17 | 30.43 |
| SE9-16 | p2pnet_bmcv.py        | p2pnet_bm1688_int8_1b.bmodel  |  20.34 | 30.71 |
| SE9-16 | p2pnet_bmcv.py        | p2pnet_bm1688_int8_4b.bmodel  |  20.34 | 30.71 |
| SE9-16 | p2pnet_bmcv.soc      | p2pnet_bm1688_fp32_1b.bmodel  |  18.06 | 28.67 |
| SE9-16 | p2pnet_bmcv.soc      | p2pnet_bm1688_fp16_1b.bmodel  |  18.15 | 28.72 |
| SE9-16 | p2pnet_bmcv.soc      | p2pnet_bm1688_int8_1b.bmodel  |  18.10 | 28.53 |
| SE9-16 | p2pnet_bmcv.soc      | p2pnet_bm1688_int8_4b.bmodel  |  18.10 | 28.53 |
| SE9-8| p2pnet_opencv.py | p2pnet_cv186x_fp32_1b.bmodel | 18.35     | 29.12          |
| SE9-8| p2pnet_opencv.py | p2pnet_cv186x_fp16_1b.bmodel | 18.33      | 29.11         |
| SE9-8| p2pnet_opencv.py | p2pnet_cv186x_int8_1b.bmodel | 18.43      | 29.54          |
| SE9-8| p2pnet_opencv.py | p2pnet_cv186x_int8_4b.bmodel | 18.43       | 29.54           |
| SE9-8| p2pnet_bmcv.py   | p2pnet_cv186x_fp32_1b.bmodel | 20.15       | 30.40           |
| SE9-8| p2pnet_bmcv.py   | p2pnet_cv186x_fp16_1b.bmodel | 20.17       | 30.43           |
| SE9-8| p2pnet_bmcv.py   | p2pnet_cv186x_int8_1b.bmodel | 20.36       | 30.72          |
| SE9-8| p2pnet_bmcv.py   | p2pnet_cv186x_int8_4b.bmodel | 20.36       | 30.72           |
| SE9-8| p2pnet_bmcv.soc | p2pnet_cv186x_fp32_1b.bmodel | 18.07       | 28.67           |
| SE9-8| p2pnet_bmcv.soc | p2pnet_cv186x_fp16_1b.bmodel | 18.06      | 28.65           |
| SE9-8| p2pnet_bmcv.soc | p2pnet_cv186x_int8_1b.bmodel | 18.10          | 28.53            |
| SE9-8| p2pnet_bmcv.soc | p2pnet_cv186x_int8_4b.bmodel | 18.10       | 28.53           |


> **测试说明**：
> 1. batch_size=4和batch_size=1的模型精度一致；
> 2.由于sdk版本之间可能存在差异，实际运行结果与本表有<0.01的精度误差是正常的；
> 3.在搭载了相同TPU和SOPHONSDK的PCIe或SoC平台上，相同程序的精度一致，SE5系列对应BM1684，SE7系列对应BM1684X，SE9系列中，SE9-16对应BM1688，SE9-8对应CV186X；

## 7. 性能测试
### 7.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684/p2pnet_bm1684_fp32_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

|                  测试模型                   | calculate time(ms) |
| ------------------------------------------ | ------------------ |
| BM1684/p2pnet_bm1684_fp32_1b.bmodel  		 | 92.4               |
| BM1684/p2pnet_bm1684_int8_1b.bmodel  		 | 47.8               |
| BM1684/p2pnet_bm1684_int8_4b.bmodel  		 | 			13.7      |
| BM1684X/p2pnet_bm1684x_fp32_1b.bmodel 	 | 			158.5     |
| BM1684X/p2pnet_bm1684x_fp16_1b.bmodel 	 | 			14.0      |
| BM1684X/p2pnet_bm1684x_int8_1b.bmodel 	 | 			7.0       |
| BM1684X/p2pnet_bm1684x_int8_4b.bmodel 	 | 			6.6       |
| BM1688/p2pnet_bm1688_fp32_1b.bmodel 	     | 			937.3     |
| BM1688/p2pnet_bm1688_fp16_1b.bmodel 	     | 			109.5     |
| BM1688/p2pnet_bm1688_int8_1b.bmodel 	     | 			28.6      |
| BM1688/p2pnet_bm1688_int8_4b.bmodel 	     | 			27.68     |
| CV186X/p2pnet_cv186x_fp32_1b.bmodel  		 | 937.7              |
| CV186X/p2pnet_cv186x_fp16_1b.bmodel  		 | 109.5              |
| CV186X/p2pnet_cv186x_int8_1b.bmodel  		 | 28.6               |
| CV186X/p2pnet_cv186x_int8_4b.bmodel  		 | 27.68              |

> **测试说明**：
> 1. 性能测试结果具有一定的波动性；
> 2. `calculate time`已折算为平均每张图片的推理时间；
> 3.SoC和PCIe的测试结果基本一致。

### 7.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++和Python例程打印的时间已经折算为单张图片的处理时间。

在不同的测试平台上，使用不同的例程、模型测试`datasets/test/images`，性能测试结果如下：
|    测试平台  |     测试程序      |             测试模型       |decode_time |preprocess_time|inference_time|postprocess_time|
| ----------- | ---------------- | -------------------------| ---------  | ------------- | ------------ | --------- |
| SE5-16 | p2pnet_opencv.py | p2pnet_bm1684_fp32_1b.bmodel  | 31.35      | 38.94         | 95.00        | 4.36      |
| SE5-16 | p2pnet_opencv.py | p2pnet_bm1684_int8_1b.bmodel  | 31.43      | 38.78         | 50.79        | 4.35      |
| SE5-16 | p2pnet_opencv.py | p2pnet_bm1684_int8_4b.bmodel  | 31.44      | 42.87         | 15.95        | 4.18      |
| SE5-16 | p2pnet_bmcv.py   | p2pnet_bm1684_fp32_1b.bmodel  | 3.40       | 3.22          | 92.91        | 4.43      |
| SE5-16 | p2pnet_bmcv.py   | p2pnet_bm1684_int8_1b.bmodel  | 3.40       | 3.21          | 48.62        | 4.42      |
| SE5-16 | p2pnet_bmcv.py   | p2pnet_bm1684_int8_4b.bmodel  | 3.08       | 3.61          | 14.09        | 4.26      |
| SE5-16 | p2pnet_bmcv.soc  | p2pnet_bm1684_fp32_1b.bmodel  | 4.414      | 0.759         | 91.985       | 1.358     |
| SE5-16 | p2pnet_bmcv.soc  | p2pnet_bm1684_int8_1b.bmodel  | 4.467      | 0.768         | 47.755       | 1.355     |
| SE5-16 | p2pnet_bmcv.soc  | p2pnet_bm1684_int8_4b.bmodel  | 4.23       | 2.789         | 54.913       | 1.339     |
| SE7-32 | p2pnet_opencv.py | p2pnet_bm1684x_fp32_1b.bmodel | 27.72      | 41.47         | 169.78       | 4.36      |
| SE7-32 | p2pnet_opencv.py | p2pnet_bm1684x_fp16_1b.bmodel | 27.74      | 40.82         | 17.85        | 4.36      |
| SE7-32 | p2pnet_opencv.py | p2pnet_bm1684x_int8_1b.bmodel | 27.72      | 41.65         | 10.47        | 4.36      |
| SE7-32 | p2pnet_opencv.py | p2pnet_bm1684x_int8_4b.bmodel | 27.80      | 44.00         | 9.36         | 4.18      |
| SE7-32 | p2pnet_bmcv.py   | p2pnet_bm1684x_fp32_1b.bmodel | 2.87       | 2.71          | 167.31       | 4.42      |
| SE7-32 | p2pnet_bmcv.py   | p2pnet_bm1684x_fp16_1b.bmodel | 2.88       | 2.72          | 15.37        | 4.43      |
| SE7-32 | p2pnet_bmcv.py   | p2pnet_bm1684x_int8_1b.bmodel | 2.87       | 2.69          | 7.98         | 4.43      |
| SE7-32 | p2pnet_bmcv.py   | p2pnet_bm1684x_int8_4b.bmodel | 2.60       | 3.04          | 7.28         | 4.27      |
| SE7-32 | p2pnet_bmcv.soc  | p2pnet_bm1684x_fp32_1b.bmodel | 3.926      | 0.827         | 166.5        | 1.358     |
| SE7-32 | p2pnet_bmcv.soc  | p2pnet_bm1684x_fp16_1b.bmodel | 3.979      | 0.827         | 14.568       | 1.356     |
| SE7-32 | p2pnet_bmcv.soc  | p2pnet_bm1684x_int8_1b.bmodel | 4.01       | 0.828         | 7.217        | 1.355     |
| SE7-32 | p2pnet_bmcv.soc  | p2pnet_bm1684x_int8_4b.bmodel | 3.859      | 0.794         | 27.76        | 1.345     |
| SE9-16 | p2pnet_opencv.py | p2pnet_bm1688_fp32_1b.bmodel  | 38.52      | 52.18         | 941.72       | 6.11      |
| SE9-16 | p2pnet_opencv.py | p2pnet_bm1688_fp16_1b.bmodel  | 38.50      | 52.08         | 113.67       | 6.15      |
| SE9-16 | p2pnet_opencv.py | p2pnet_bm1688_int8_1b.bmodel  | 38.57      | 51.91         | 32.77        | 6.10      |
| SE9-16 | p2pnet_opencv.py | p2pnet_bm1688_int8_4b.bmodel  | 38.67      | 56.83         | 30.89        | 5.83      |
| SE9-16 | p2pnet_bmcv.py   | p2pnet_bm1688_fp32_1b.bmodel  | 5.32       | 5.91          | 938.84       | 6.27      |
| SE9-16 | p2pnet_bmcv.py   | p2pnet_bm1688_fp16_1b.bmodel  | 5.31       | 5.92          | 110.76       | 6.24      |
| SE9-16 | p2pnet_bmcv.py   | p2pnet_bm1688_int8_1b.bmodel  | 5.31       | 5.91          | 29.79        | 6.23      |
| SE9-16 | p2pnet_bmcv.py   | p2pnet_bm1688_int8_4b.bmodel  | 5.05       | 6.26          | 28.25        | 5.95      |
| SE9-16 | p2pnet_bmcv.soc  | p2pnet_bm1688_fp32_1b.bmodel  | 6.307      | 1.968         | 937.301      | 1.973     |
| SE9-16 | p2pnet_bmcv.soc  | p2pnet_bm1688_fp16_1b.bmodel  | 6.357      | 1.966         | 109.443      | 1.936     |
| SE9-16 | p2pnet_bmcv.soc  | p2pnet_bm1688_int8_1b.bmodel  | 6.456      | 1.965         | 28.535       | 1.934     |
| SE9-16 | p2pnet_bmcv.soc  | p2pnet_bm1688_int8_4b.bmodel  | 6.268      | 1.887         | 110.907      | 1.97     |
| SE9-8  | p2pnet_opencv.py | p2pnet_cv186x_fp32_1b.bmodel  | 39.86      | 53.21         | 942.04       | 6.12      |
| SE9-8  | p2pnet_opencv.py | p2pnet_cv186x_fp16_1b.bmodel  | 38.53      | 52.53         | 113.74       | 6.12      |
| SE9-8  | p2pnet_opencv.py | p2pnet_cv186x_int8_1b.bmodel  | 38.52      | 52.45         | 32.81        | 6.11      |
| SE9-8  | p2pnet_opencv.py | p2pnet_cv186x_int8_4b.bmodel  | 38.66      | 57.69         | 30.72        | 5.85      |
| SE9-8  | p2pnet_bmcv.py   | p2pnet_cv186x_fp32_1b.bmodel  | 9.01       | 5.80          | 939.25       | 6.23      |
| SE9-8  | p2pnet_bmcv.py   | p2pnet_cv186x_fp16_1b.bmodel  | 5.41       | 5.82          | 110.8        | 6.23      |
| SE9-8  | p2pnet_bmcv.py   | p2pnet_cv186x_int8_1b.bmodel  | 5.22       | 5.80          | 29.82        | 6.24      |
| SE9-8  | p2pnet_bmcv.py   | p2pnet_cv186x_int8_4b.bmodel  | 4.88       | 6.19          | 28.18        | 6.01      |
| SE9-8  | p2pnet_bmcv.soc  | p2pnet_cv186x_fp32_1b.bmodel  | 6.237      | 1.977         | 937.621      | 1.972     |
| SE9-8  | p2pnet_bmcv.soc  | p2pnet_cv186x_fp16_1b.bmodel  | 6.358      | 1.976         | 109.408      | 1.935     |
| SE9-8  | p2pnet_bmcv.soc  | p2pnet_cv186x_int8_1b.bmodel  | 6.334      | 1.972         | 28.505       | 1.917     |
| SE9-8  | p2pnet_bmcv.soc  | p2pnet_cv186x_int8_4b.bmodel  | 6.188      | 1.88          | 110.584      | 1.89      |


> **测试说明**：
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. SE5-16/SE7-32的主控处理器均为8核CA53@2.3GHz，SE9-16为8核CA53@1.6GHz，SE9-8为6核CA53@1.6GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异，不同的阈值对后处理时间影响较大。

## 8. FAQ
其他问题请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。