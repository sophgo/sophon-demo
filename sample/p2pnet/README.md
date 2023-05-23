# P2PNet
## 目录

 * [1. 简介](#1-简介)
 * [2. 特性](#2-特性)
 * [3. 准备模型与数据](#3-准备模型与数据)
 * [4. 模型编译](#4-模型编译)
   * [4.1 TPU-NNTC编译BModel](#41-TPU-NNTC编译BModel)
   * [4.2 TPU-MLIR编译BModel](#42-TPU-MLIR编译BModel)
 * [5. 例程测试](#5-例程测试)
 * [6. 精度测试](#5-精度测试)
   * [6.1 测试方法](#61-测试方法)
   * [6.2 测试结果](#61-测试结果)
* [7. 性能测试](#7-性能测试)
   * [7.1 bmrt_test](#71-bmrt_test)
   * [7.2 程序运行性能](#72-程序运行性能)
* [8. FAQ](#8-FAQ)
 


## 1. 简介
P2PNet是腾讯优图实验室提出了点对点网络（Point-to-Point Network，P2PNet），业界首创直接预测人头中心点的人群计数新范式，能够同时实现人群个体定位和人群计数，该算法在 2020 年 12 月份刷新 NWPU 榜单。

**论文地址**：https://arxiv.org/pdf/2107.12746.pdf

**项目地址**：https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet

**数据集**:https://www.datafountain.cn/datasets/5670，该数据集是一个可用于图像密集人群计数的数据集，分为PartA和PartB：
PartA： 共计482张图片，其中训练集300张，测试集182张；
PartB： 共计716张图片，其中训练集400张，测试集316张。

## 2. 特性
* 支持BM1684X(x86 PCIe、SoC)和BM1684(x86 PCIe、SoC、arm PCIe)
* 支持FP32、FP16(BM1684X)、INT8模型编译和推理
* 支持基于BMCV预处理的C++推理
* 支持基于OpenCV和BMCV预处理的Python推理
* 支持单batch和多batch模型推理
* 支持图片和视频测试

## 3. 准备模型与数据

​Pytorch的模型在编译前要经过`torch.jit.trace`，trace后的模型才能用于编译BModel，trace方法可以参考官方`export.py`。

​同时，您需要准备用于测试的数据集，如果量化模型，还要准备用于量化的数据集。

​本例程在`scripts`目录下提供了相关模型和数据（后续demo会使用）的下载脚本`download.sh`，您也可以自己准备模型和数据集，并参考[4. 模型编译](#4-模型编译)进行模型转换。

```bash
cd scripts
chmod +x ./*
./download.sh
```

​	执行后，模型保存至`models/`，数据集下载并解压至`datasets/`

下载的模型包括：
```
./models
├── BM1684
│   ├──p2pnet_bm1684_fp32_1b.bmodel   # 使用TPU-NNTC编译，用于BM1684的FP32 BModel，batch_size=1
│   ├──p2pnet_bm1684_int8_1b.bmodel   # 使用TPU-NNTC编译，用于BM1684的INT8 BModel，batch_size=1
│   └──p2pnet_bm1684_int8_4b.bmodel   # 使用TPU-NNTC编译，用于BM1684的INT8 BModel，batch_size=4
├── BM1684X
│   ├──p2pnet_bm1684x_fp32_1b.bmodel  # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=1
│   ├──p2pnet_bm1684x_fp16_1b.bmodel  # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=1
│   ├──p2pnet_bm1684x_int8_1b.bmodel  # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=1
│   └──p2pnet_bm1684x_int8_4b.bmodel  # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=4
│── onnx
│   ├──p2pnet_1b.onnx				  # onnx模型，batch_size=1
│   └──p2pnet_4b.onnx				  # onnx模型，batch_size=4
│── torch
│   ├──p2pnet_trace.pt				  # trace后的torchscript模型
│   └──p2pnet.pth					  # pytorch模型
```
下载的数据包括：
```
./datasets
├── ShanghaiTech
│   ├── ShanghaiTech-Dataset
│       ├── ShanghaiTech
│           ├── part_A
│           ├── part_B
├── video
│   ├── video.avi
```

​模型信息：

| 模型名称 | [P2PNet](https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet/blob/main/weights/SHTechA.pth) |
| :------- | :----------------------------------------------------------- |
| 训练集   | Shanghaitech dataset                                         |
| 预处理   | RGB planar, mean[123.675,116.28,103.53], scale[0.01712,0.01751,0.01743]|
| 输入数据 | images, [batch_size, 3, 512, 512], FP32，NCHW，RGB planar    |
| 输出数据 | 53, [batch_size, 16384, 2], FP32 <br />55, [batch_size, 16384, 2], FP32 |                                                     |

## 4. 模型编译

导出的模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。如果您使用BM1684芯片，建议使用TPU-NNTC编译BModel；如果您使用BM1684X芯片，建议使用TPU-MLIR编译BModel。
### 4.1 TPU-NNTC编译BModel
- 生成FP32 BModel

​本例程在`scripts`目录下提供了编译FP32 BModel的脚本。请注意修改`gen_fp32bmodel_nntc.sh`中的JIT模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（支持BM1684和BM1684X），如：

```bash
cd scripts
chmod +x ./*
./scripts/gen_fp32bmodel_nntc.sh BM1684
```

​执行上述命令会在`models/BM1684`下生成`p2pnet_bm1684_fp32_1b.bmodel`文件，即转换好的FP32 BModel。

- 生成INT8 BModel

​不量化模型可跳过本节。

​本例程在`scripts`目录下提供了量化INT8 BModel的脚本。请注意修改`gen_int8bmodel_nntc.sh`中的JIT模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台（支持BM1684和BM1684X），如：

```shell
cd scripts
chmod +x ./*
./scripts/gen_int8bmodel_nntc.sh BM1684
```

​上述脚本会在`models/BM1684`下生成`p2pnet_bm1684_int8_1b.bmodel`文件，即转换好的INT8 BModel。
### 4.2 TPU-MLIR编译BModel
模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#2-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index/material/31/all.html)相应版本的SDK中获取)。

- 生成FP32 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（支持BM1684X），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684x
```

​执行上述命令会在`models/BM1684X/`下生成`p2pnet_bm1684x_fp32_1b.bmodel`文件，即转换好的FP32 BModel。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（支持BM1684X），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x
```

​执行上述命令会在`models/BM1684X/`下生成`p2pnet_bm1684x_fp16_1b.bmodel`文件，即转换好的FP16 BModel。

- 生成INT8 BModel

​本例程在`scripts`目录下提供了量化INT8 BModel的脚本，请注意修改`gen_int8bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台（支持BM1684X），如：

```shell
./scripts/gen_int8bmodel_mlir.sh bm1684x
```

​上述脚本会在`models/BM1684X`下生成`p2pnet_bm1684x_int8_1b.bmodel`等文件，即转换好的INT8 BModel。

## 5. 例程测试
- [C++例程](./cpp/README.md)
- [Python例程](./python/README.md)

## 6. 精度测试
### 6.1 测试方法

首先，参考[C++例程](cpp/README.md#69-测试数据集)或[Python例程](python/README.md#47-测试数据集)推理要测试的数据集，生成预测的txt文件。
然后，使用`tools`目录下的`eval_SHTech.py`脚本，将测试生成的txt文件与测试集标签mat文件进行对比，计算出准确率信息，命令如下：
```bash
# 请根据实际情况修改groundtruth和测试结果路径
python3 tools/eval_SHTech.py --gt_path datasets/ground_truth_path/ --result_path cpp/p2pnet_bmcv/results/
```
### 6.2 测试结果
根据本例程提供的数据集，测试结果如下：
|   测试平台   |        测试程序       |         	 测试模型       	| MAE | MSE |
| ------------ | --------------------- | ------------------------------ | --- | --- |
| pytorch      | p2pnet_trace_pt.py    | p2pnet_trace.pt         		|  84 | 147 |
| BM1684 soc   | p2pnet_opencv.py      | p2pnet_bm1684_fp32_1b.bmodel   |  84 | 147 |
| BM1684 soc   | p2pnet_opencv.py      | p2pnet_bm1684_int8_1b.bmodel   |  84 | 141 |
| BM1684 soc   | p2pnet_bmcv.py        | p2pnet_bm1684_fp32_1b.bmodel   |  96 | 174 |
| BM1684 soc   | p2pnet_bmcv.py        | p2pnet_bm1684_int8_1b.bmodel   |  94 | 157 |
| BM1684 soc   | p2pnet_bmcv.soc       | p2pnet_bm1684_fp32_1b.bmodel   |  81 | 144 |
| BM1684 soc   | p2pnet_bmcv.soc       | p2pnet_bm1684_int8_1b.bmodel   | 113 | 193 |
| onnx         | p2pnet_onnx.py        | p2pnet_1b.onnx                 |  84 | 147 |
| BM1684X pcie | p2pnet_opencv.py      | p2pnet_bm1684x_fp32_1b.bmodel  |  84 | 148 |
| BM1684X pcie | p2pnet_bmcv.py        | p2pnet_bm1684x_fp16_1b.bmodel  |  98 | 173 |
| BM1684X pcie | p2pnet_bmcv.py        | p2pnet_bm1684x_int8_1b.bmodel  |  96 | 170 |

> **测试说明**：  
> 1. 由于opencv和bmcv解码存在差异，相同模型的准确率也会存在细微差异；

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
| ------------------------------------------- | ------------------ |
| BM1684/p2pnet_bm1684_fp32_1b.bmodel  		  |			171.2      |
| BM1684/p2pnet_bm1684_int8_1b.bmodel  		  | 		5.7        |
| BM1684/p2pnet_bm1684_int8_4b.bmodel  		  |			5.7        |
| BM1684X/p2pnet_bm1684x_fp32_1b.bmodel 	  |			160.4      |
| BM1684X/p2pnet_bm1684x_fp16_1b.bmodel 	  |			14.4       |
| BM1684X/p2pnet_bm1684x_int8_1b.bmodel 	  |			7.3        |
| BM1684X/p2pnet_bm1684x_int8_4b.bmodel 	  |			4.8        |

> **测试说明**：  
> 1. 性能测试结果具有一定的波动性；
> 2. `calculate time`已折算为平均每张图片的推理时间；

### 7.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++例程打印的预处理时间、推理时间、后处理时间为整个batch处理的时间，需除以相应的batch size才是每张图片的处理时间。

在不同的测试平台上，使用不同的例程、模型测试`datasets/video/video.avi`，性能测试结果如下：
|    测试平台  |     测试程序      |             测试模型        |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ---------------- | ----------------------------- | --------- | ------------- | ------------ | --------- |
| BM1684 pcie | p2pnet_opencv.py | p2pnet_bm1684_fp32_1b.bmodel  | 8.4       | 15.1          | 174.1        | 3.7       |
| BM1684 pcie | p2pnet_opencv.py | p2pnet_bm1684_int8_1b.bmodel  | 8.2       | 15.5          | 62.3         | 3.1       |
| BM1684 pcie | p2pnet_opencv.py | p2pnet_bm1684_int8_4b.bmodel  | 8.1       | 15.4          | 62.2         | 3.1       |
| BM1684 pcie | p2pnet_bmcv.py   | p2pnet_bm1684_fp32_1b.bmodel  | 9.3       | 2.9           | 174.7        | 3.7       |
| BM1684 pcie | p2pnet_bmcv.py   | p2pnet_bm1684_int8_1b.bmodel  | 9.3       | 2.8           | 58.9         | 6.1       |
| BM1684 pcie | p2pnet_bmcv.py   | p2pnet_bm1684_int8_4b.bmodel  | 9.3       | 2.8           | 59.0         | 6.1       |
| BM1684X pcie| p2pnet_opencv.py | p2pnet_bm1684x_fp32_1b.bmodel | 5.2       | 7.5           | 162.9        | 5.4       |
| BM1684X pcie| p2pnet_opencv.py | p2pnet_bm1684x_fp16_1b.bmodel | 3.8       | 6.5           | 16.4         | 1.1       |
| BM1684X pcie| p2pnet_opencv.py | p2pnet_bm1684x_int8_1b.bmodel | 3.8       | 6.4           | 9.2          | 1.0       |
| BM1684X pcie| p2pnet_opencv.py | p2pnet_bm1684x_int8_4b.bmodel | 3.8       | 6.4           | 9.2          | 1.0       |
| BM1684X pcie| p2pnet_bmcv.py   | p2pnet_bm1684x_fp32_1b.bmodel | 18.0      | 3.4           | 165.3        | 5.4       |
| BM1684X pcie| p2pnet_bmcv.py   | p2pnet_bm1684x_fp16_1b.bmodel | 18.1      | 3.4           | 19.4         | 5.2       |
| BM1684X pcie| p2pnet_bmcv.py   | p2pnet_bm1684x_int8_1b.bmodel | 17.9      | 3.4           | 12.1         | 5.2       |
| BM1684X pcie| p2pnet_bmcv.py   | p2pnet_bm1684x_int8_4b.bmodel | 18.0      | 3.4           | 12.2         | 5.2       |

> **测试说明**：  
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. 图像分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异。

## 8. FAQ
请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。