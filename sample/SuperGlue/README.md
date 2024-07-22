# SuperGlue

* [1. 简介](#1-简介)
* [2. 特性](#2-特性)
* [3. 准备模型、数据、依赖库](#3-准备模型数据依赖库)
* [4. 模型编译](#4-模型编译)
* [5. 例程测试](#5-例程测试)
* [6. 精度测试](#6-精度测试)
  * [6.1 测试方法](#61-测试方法)
  * [6.2 测试结果](#62-测试结果)
* [7. 性能测试](#7-性能测试)
  * [7.1 bmrt_test](#71-bmrt_test)
  * [7.2 程序运行性能](#72-程序运行性能)
* [8. FAQ](#8-faq)

## 目录

## 1. 简介

SuperGlue是Magic Leap完成的CVPR 2020研究项目。SuperGlue网络是一个图神经网络，结合了一个最优匹配层，该层经过训练，可以对两组稀疏图像特征执行匹配。本例程对[SuperGlue官方开源仓库](https://github.com/magicleap/SuperGluePretrainedNetwork)的模型和算法进行移植，使之能在SOPHON BM1684X/BM1688/CV186X上进行推理测试。

## 2. 特性

* 支持BM1688/CV186X(SoC)、BM1684X(x86 PCIe、SoC)
* 支持FP32、FP16(BM1684X/BM1688/CV186X)模型编译和推理
* 支持基于OpenCV解码、BMCV预处理、BMRT推理、LIBTORCH后处理的C++推理
* 支持单batch模型

## 3. 准备模型、数据、依赖库

​本例程在`scripts`目录下提供了本例程所需的相关模型、数据、依赖库的下载脚本`download.sh`，您也可以自己准备模型和数据集。

```bash
chmod -R +x scripts/
./scripts/download.sh
```
下载的模型包括：

```bash
./models
├── BM1684X
│   ├── superglue_fp16_1b_iter20_1024.bmodel # 使用TPU-MLIR编译，用于BM1684X的superglue FP16 BModel，batch_size=1，sinkhorn_iterations=20，max_keypoint_size=1024
│   ├── superglue_fp32_1b_iter20_1024.bmodel # 使用TPU-MLIR编译，用于BM1684X的superglue FP32 BModel，batch_size=1，sinkhorn_iterations=20，max_keypoint_size=1024
│   ├── superpoint_fp16_1b.bmodel            # 使用TPU-MLIR编译，用于BM1684X的superpoint FP16 BModel，batch_size=1
│   └── superpoint_fp32_1b.bmodel            # 使用TPU-MLIR编译，用于BM1684X的superpoint FP32 BModel，batch_size=1
├── BM1688
│   ├── superglue_fp16_1b_iter20_1024.bmodel # 使用TPU-MLIR编译，用于BM1688的superglue FP16 BModel，batch_size=1，sinkhorn_iterations=20，max_keypoint_size=1024
│   ├── superglue_fp32_1b_iter20_1024.bmodel # 使用TPU-MLIR编译，用于BM1688的superglue FP32 BModel，batch_size=1，sinkhorn_iterations=20，max_keypoint_size=1024
│   ├── superpoint_fp16_1b.bmodel            # 使用TPU-MLIR编译，用于BM1688的superpoint FP16 BModel，batch_size=1
│   └── superpoint_fp32_1b.bmodel            # 使用TPU-MLIR编译，用于BM1688的superpoint FP32 BModel，batch_size=1
├── CV186X
│   ├── superglue_fp16_1b_iter20_1024.bmodel # 使用TPU-MLIR编译，用于CV186X的superglue FP16 BModel，batch_size=1，sinkhorn_iterations=20，max_keypoint_size=1024
│   ├── superglue_fp32_1b_iter20_1024.bmodel # 使用TPU-MLIR编译，用于CV186X的superglue FP32 BModel，batch_size=1，sinkhorn_iterations=20，max_keypoint_size=1024
│   ├── superpoint_fp16_1b.bmodel            # 使用TPU-MLIR编译，用于CV186X的superpoint FP16 BModel，batch_size=1
│   └── superpoint_fp32_1b.bmodel            # 使用TPU-MLIR编译，用于CV186X的superpoint FP32 BModel，batch_size=1
├── onnx
│   ├── superglue_indoor_iter20_1024.onnx    # superglue的onnx模型
│   ├── superpoint_fp16_qtable               # superpoint编译fp16精度bmodel所需的敏感层
│   └── superpoint_to_nms.onnx               # superpoint的onnx模型
└── weights                                  # 源仓库的权重
    ├── superglue_indoor.pth
    ├── superglue_outdoor.pth
    └── superpoint_v1.pth
```

下载的数据包括：

```bash
./datasets
├── scannet_sample_images               #测试数据集
├── scannet_sample_pairs_with_gt.txt    #测试数据集标签文件
├── superglue_test_input                #superglue的测试输入，供mlir模型编译精度对齐使用
└── superpoint_test_input               #superpoint的测试输入，供mlir模型编译精度对齐使用
```

下载的依赖库包括：

```bash
./cpp/
├── aarch64_lib                         # 交叉编译依赖库，包含aarch64的libtorch以及其他第三方库
└── libtorch                            # x86的libtorch依赖库
```
## 4. 模型编译

Pytorch模型在编译前要导出成onnx模型，如果您希望自己导出模型，可参考[SuperGlue模型导出](./docs/SuperGlue_Export_Guide.md)。

模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index/material/all/all.html)相应版本的SDK中获取)。

- 生成FP32 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684x #bm1688/cv186x
```

​执行上述命令会在`models/BM1684X`等文件夹下生成`superpoint_fp32_1b.bmodel`和`superglue_fp32_1b_iter20_1024.bmodel`文件，即转换好的FP32 BModel。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x #bm1688/cv186x
```

​执行上述命令会在`models/BM1684X/`等文件夹下生成`superpoint_fp16_1b.bmodel`和`superglue_fp16_1b_iter20_1024.bmodel`文件，即转换好的FP16 BModel。

## 5. 例程测试
- [C++例程](./cpp/README.md)

## 6. 精度测试
### 6.1 测试方法

首先，参考[C++例程](cpp/README.md#32-测试图片)推理要测试的数据集(scannet_sample_images)，生成预测的json文件。  
然后，使用`tools`目录下的`eval.py`脚本，将测试生成的json文件与测试集标签文件进行对比，计算出superglue的MScore评价指标，命令如下：
```bash
# 安装opencv，若已安装请跳过
pip3 install opencv-python-headless matplotlib
# 请根据实际情况修改程序路径和json文件路径
cd tools
python3 eval.py --input_pairs ../datasets/scannet_sample_pairs_with_gt.txt --result_json ../cpp/superglue_bmcv/results/result.json
```

### 6.2 测试结果

| 测试平台      |  测试程序        |  superpoint模型  |   superglue模型  |MScore|
| ------------ | ---------------- | ----------------| ---------------  | ----- |
| SE7-32       | superglue_bmcv.soc | superpoint_fp32_1b.bmodel | superglue_fp32_1b_iter20_1024.bmodel |    16.90 |
| SE7-32       | superglue_bmcv.soc | superpoint_fp16_1b.bmodel | superglue_fp16_1b_iter20_1024.bmodel |    16.69 |
| SE9-16       | superglue_bmcv.soc | superpoint_fp32_1b.bmodel | superglue_fp32_1b_iter20_1024.bmodel |    16.90 |
| SE9-16       | superglue_bmcv.soc | superpoint_fp16_1b.bmodel | superglue_fp16_1b_iter20_1024.bmodel |    16.71 |
| SE9-8        | superglue_bmcv.soc | superpoint_fp32_1b.bmodel | superglue_fp32_1b_iter20_1024.bmodel |    16.90 |
| SE9-8        | superglue_bmcv.soc | superpoint_fp16_1b.bmodel | superglue_fp16_1b_iter20_1024.bmodel |    16.71 |

> **测试说明**：  
> 1. 由于sdk版本之间可能存在差异，实际运行结果与本表有<0.01的精度误差是正常的；
> 2. 在搭载了相同TPU和SOPHONSDK的PCIe或SoC平台上，相同程序的精度一致，SE5系列对应BM1684，SE7系列对应BM1684X，SE9系列中，SE9-16对应BM1688，SE9-8对应CV186X；

## 7. 性能测试
### 7.1 bmrt_test

使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684X/superpoint_fp32_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

|                  测试模型                         | calculate time(ms) |
| -------------------------------------------       | ----------------- |
| BM1684X/superpoint_fp32_1b.bmodel            |          51.46  |
| BM1684X/superpoint_fp16_1b.bmodel            |          10.76  |
| BM1684X/superglue_fp32_1b_iter20_1024.bmodel |         289.65  |
| BM1684X/superglue_fp16_1b_iter20_1024.bmodel |          75.51  |
| BM1688/superpoint_fp32_1b.bmodel   |         224.76  |
| BM1688/superpoint_fp16_1b.bmodel   |          41.47  |
| BM1688/superglue_fp32_1b_iter20_1024.bmodel|         670.28  |
| BM1688/superglue_fp16_1b_iter20_1024.bmodel|         182.35  |
| CV186X/superpoint_fp32_1b.bmodel   |         224.76  |
| CV186X/superpoint_fp16_1b.bmodel   |          41.51  |
| CV186X/superglue_fp32_1b_iter20_1024.bmodel|         667.09  |
| CV186X/superglue_fp16_1b_iter20_1024.bmodel|         179.36  |

> **测试说明**：  
> 1. 性能测试结果具有一定的波动性；
> 2. SoC和PCIe的测试结果基本一致。

### 7.2 程序运行性能

参考[C++例程](cpp/README.md)运行程序，并查看统计的性能信息。

|    测试平台  |     测试程序     | superpoint模型            |   superglue模型                    | decode_time    |superpoint_time  |superglue_time   | 
| ----------- | ---------------- | ---------------          | ----------------                   | --------       | ---------       | ---------     |
|   SE7-32    |superglue_bmcv.soc |superpoint_fp32_1b.bmodel|superglue_fp32_1b_iter20_1024.bmodel|      4.16       |      83.09      |     300.97      |
|   SE7-32    |superglue_bmcv.soc |superpoint_fp16_1b.bmodel|superglue_fp16_1b_iter20_1024.bmodel|      4.18       |      52.52      |      86.55      |
|   SE9-16    |superglue_bmcv.soc |superpoint_fp32_1b.bmodel|superglue_fp32_1b_iter20_1024.bmodel|      5.70       |     263.51      |     686.38      |
|   SE9-16    |superglue_bmcv.soc |superpoint_fp16_1b.bmodel|superglue_fp16_1b_iter20_1024.bmodel|      5.62       |      89.39      |     198.20      |
|    SE9-8    |superglue_bmcv.soc |superpoint_fp32_1b.bmodel|superglue_fp32_1b_iter20_1024.bmodel|      5.45       |     269.53      |     686.53      |
|    SE9-8    |superglue_bmcv.soc |superpoint_fp16_1b.bmodel|superglue_fp16_1b_iter20_1024.bmodel|      5.78       |      84.62      |     200.16      |

> **测试说明**：  
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. SE5-16/SE7-32的主控处理器均为8核CA53@2.3GHz，SE9-16为8核CA53@1.6GHz，SE9-8为6核CA53@1.6GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异，不同的阈值对后处理时间影响较大。

## 8. FAQ

请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。