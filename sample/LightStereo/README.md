[简体中文](./README.md) | [English](./README_EN.md)

# LightStereo

## 目录

- [LightStereo](#lightstereo)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
    - [2.1 目录结构说明](#21-目录结构说明)
    - [2.2 特性](#22-特性)
  - [3. 数据准备与模型编译](#3-数据准备与模型编译)
    - [3.1 数据准备](#31-数据准备)
    - [3.2 模型编译](#32-模型编译)
  - [4. 例程测试](#4-例程测试)
  - [5. 精度测试](#5-精度测试)
    - [5.1 测试方法](#51-测试方法)
    - [5.2 测试结果](#52-测试结果)
  - [6. 性能测试](#6-性能测试)
    - [6.1 bmrt\_test](#61-bmrt_test)
    - [6.2 程序运行性能](#62-程序运行性能)
  - [7. FAQ](#7-faq)
  
## 1. 简介
LightStereo是一种用于双目立体匹配的神经网络模型，它的输入是双目摄像头的左图和右图，输出是视差图（disparity map）。本例程对[LightStereo官方开源仓库](https://github.com/XiandaGuo/OpenStereo)的模型和算法进行移植，使之能在SOPHON BM1684X/BM1688/CV186X上进行推理测试。

## 2. 特性

### 2.1 目录结构说明
```bash
├── cpp                   # 存放C++例程及其README
|   ├── README.md      
|   └── lightstereo_bmcv   # 使用Sophon-OpenCV解码、BMCV前处理、BMRT推理的C++例程
├── docs                  # 存放本例程专用文档，如ONNX导出等
├── pics                  # 存放README等说明文档中用到的图片
├── python                # 存放Python例程及其README
|   ├── README_EN.md 
|   ├── README.md 
|   ├── lightstereo_bmcv.py     # 使用SAIL解码、SAIL.BMCV前处理、SAIL推理的Python例程
|   └── lightstereo_opencv.py   # 使用OpenCV解码、OpenCV前处理、SAIL推理的Python例程
├── README.md             # 本例程的中文指南
├── scripts               # 存放模型编译、数据下载、自动测试等shell脚本
└── tools                 # 存放精度测试、性能比对等python脚本
```

### 2.2 特性
* 支持BM1688/CV186X(SoC)、BM1684X(x86 PCIe、SoC)
* 支持FP32、FP16、INT8模型编译和推理
* 支持基于BMCV预处理的C++推理
* 支持基于OpenCV和BMCV预处理的Python推理
* 支持单batch和多batch模型推理
* 支持图片测试

## 3. 数据准备与模型编译

### 3.1 数据准备

​本例程在`scripts`目录下提供了相关模型和数据的下载脚本`download.sh`，**如果您希望自己准备模型和数据集，可以跳过本小节，参考[3.2 模型编译](#32-模型编译)进行模型转换。**

```bash
chmod -R +x scripts/
./scripts/download.sh --all 
```

`download.sh`默认下载`datasets`，`models`可以通过指定参数分平台下载，参数如下：
```bash
--all     # 下载所有模型
--BM1684X # 下载BM1684X的bmodel
--BM1688  # 下载BM1688的bmodel
--CV186X  # 下载CV186X的bmodel
--onnx    # 下载onnx
--ckpt    # 下载源模型权重
```

下载的模型包括：
```bash
models/
├── BM1684X
│   ├── LightStereo-S-SceneFlow_fp16_1b.bmodel
│   └── LightStereo-S-SceneFlow_fp32_1b.bmodel
├── BM1688
│   ├── LightStereo-S-SceneFlow_fp16_1b_2core.bmodel
│   ├── LightStereo-S-SceneFlow_fp16_1b.bmodel
│   ├── LightStereo-S-SceneFlow_fp32_1b_2core.bmodel
│   └── LightStereo-S-SceneFlow_fp32_1b.bmodel
├── ckpt
│   └── LightStereo-S-SceneFlow.ckpt
├── CV186X
│   ├── LightStereo-S-SceneFlow_fp16_1b.bmodel
│   └── LightStereo-S-SceneFlow_fp32_1b.bmodel
└── onnx
    └── LightStereo-S-SceneFlow.onnx       
```
下载的数据包括：
```bash
datasets/
├── cali_data   # 量化数据集（目前量化暂时有精度问题）
└── KITTI12     # KITTI12数据集(为了节省空间，这里只放了部分训练集)
```

### 3.2 模型编译

**如果您不编译模型，只想直接使用下载的数据集和模型，可以跳过本小节。**

源模型需要编译成BModel才能在SOPHON TPU上运行，源模型在编译前要导出成onnx模型，如果您使用的TPU-MLIR版本>=v1.3.0（即官网v23.07.01），也可以直接使用torchscript模型。具体可参考[LightStereo模型导出](./docs/LightStereo_Export_Guide.md)。​同时，您需要准备用于测试的数据集，如果量化模型，还要准备用于量化的数据集。

建议使用TPU-MLIR编译BModel，模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录，并使用本例程提供的脚本将onnx模型编译为BModel。脚本中命令的详细说明可参考《TPU-MLIR开发手册》(请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的SDK中获取)。

- 生成FP32 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684x #bm1688/cv186x
```

​执行上述命令会在`models/BM1684X`等文件夹下生成转换好的FP32 BModel。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x #bm1688/cv186x
```

​执行上述命令会在`models/BM1684X/`等文件夹下生成转换好的FP16 BModel。

- 生成INT8 BModel

​本例程在`scripts`目录下提供了量化INT8 BModel的脚本，请注意修改`gen_int8bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台（**支持BM1684X/BM1688/CV186X**），如：

```shell
./scripts/gen_int8bmodel_mlir.sh bm1684x #bm1688/cv186x
```

​上述脚本会在`models/BM1684`等文件夹下生成转换好的INT8 BModel。


## 4. 例程测试
- [C++例程](./cpp/README.md)
- [Python例程](./python/README.md)

## 5. 精度测试
### 5.1 测试方法

首先，参考[C++例程](cpp/README.md#32-测试图片)或[Python例程](python/README.md#22-测试图片)推理要测试的数据集（datasets/KITTI12/kitti12_train194.txt），生成结果图片文件夹。 
然后，使用`tools`目录下的`eval.py`脚本，计算出立体匹配的评价指标，命令如下：
```bash
# 请根据实际情况修改路径
cd tools
python3 eval.py --gt_path ../datasets/KITTI12/training/disp_occ --results_path ../python/results/images
```
### 5.2 测试结果
在`datasets/KITTI12`数据集上，精度测试结果如下：
|   测试平台    |      测试程序          |              测试模型               |    D1     |
| ------------ | ----------------      | ----------------------------------- | --------- |
| SE7-32       | lightstereo_opencv.py     | LightStereo-S-SceneFlow_fp32_1b.bmodel |    0.454 |
| SE7-32       | lightstereo_opencv.py     | LightStereo-S-SceneFlow_fp16_1b.bmodel |    0.453 |
| SE7-32       | lightstereo_opencv.py     | LightStereo-S-SceneFlow_int8_1b.bmodel |    0.405 |
| SE7-32       | lightstereo_opencv.py     | LightStereo-S-SceneFlow_int8_4b.bmodel |    0.405 |
| SE7-32       | lightstereo_bmcv.py       | LightStereo-S-SceneFlow_fp32_1b.bmodel |    0.457 |
| SE7-32       | lightstereo_bmcv.py       | LightStereo-S-SceneFlow_fp16_1b.bmodel |    0.456 |
| SE7-32       | lightstereo_bmcv.py       | LightStereo-S-SceneFlow_int8_1b.bmodel |    0.408 |
| SE7-32       | lightstereo_bmcv.py       | LightStereo-S-SceneFlow_int8_4b.bmodel |    0.408 |
| SE7-32       | lightstereo_bmcv.soc      | LightStereo-S-SceneFlow_fp32_1b.bmodel |    0.457 |
| SE7-32       | lightstereo_bmcv.soc      | LightStereo-S-SceneFlow_fp16_1b.bmodel |    0.456 |
| SE7-32       | lightstereo_bmcv.soc      | LightStereo-S-SceneFlow_int8_1b.bmodel |    0.408 |
| SE7-32       | lightstereo_bmcv.soc      | LightStereo-S-SceneFlow_int8_4b.bmodel |    0.408 |
| SE9-16       | lightstereo_opencv.py     | LightStereo-S-SceneFlow_fp32_1b.bmodel |    0.454 |
| SE9-16       | lightstereo_opencv.py     | LightStereo-S-SceneFlow_fp16_1b.bmodel |    0.453 |
| SE9-16       | lightstereo_opencv.py     | LightStereo-S-SceneFlow_int8_1b.bmodel |    0.405 |
| SE9-16       | lightstereo_opencv.py     | LightStereo-S-SceneFlow_int8_4b.bmodel |    0.405 |
| SE9-16       | lightstereo_bmcv.py       | LightStereo-S-SceneFlow_fp32_1b.bmodel |    0.457 |
| SE9-16       | lightstereo_bmcv.py       | LightStereo-S-SceneFlow_fp16_1b.bmodel |    0.456 |
| SE9-16       | lightstereo_bmcv.py       | LightStereo-S-SceneFlow_int8_1b.bmodel |    0.412 |
| SE9-16       | lightstereo_bmcv.py       | LightStereo-S-SceneFlow_int8_4b.bmodel |    0.412 |
| SE9-16       | lightstereo_bmcv.soc      | LightStereo-S-SceneFlow_fp32_1b.bmodel |    0.457 |
| SE9-16       | lightstereo_bmcv.soc      | LightStereo-S-SceneFlow_fp16_1b.bmodel |    0.456 |
| SE9-16       | lightstereo_bmcv.soc      | LightStereo-S-SceneFlow_int8_1b.bmodel |    0.412 |
| SE9-16       | lightstereo_bmcv.soc      | LightStereo-S-SceneFlow_int8_4b.bmodel |    0.412 |
| SE9-16       | lightstereo_opencv.py     | LightStereo-S-SceneFlow_fp32_1b_2core.bmodel |    0.454 |
| SE9-16       | lightstereo_opencv.py     | LightStereo-S-SceneFlow_fp16_1b_2core.bmodel |    0.453 |
| SE9-16       | lightstereo_opencv.py     | LightStereo-S-SceneFlow_int8_1b_2core.bmodel |    0.405 |
| SE9-16       | lightstereo_opencv.py     | LightStereo-S-SceneFlow_int8_4b_2core.bmodel |    0.405 |
| SE9-16       | lightstereo_bmcv.py       | LightStereo-S-SceneFlow_fp32_1b_2core.bmodel |    0.457 |
| SE9-16       | lightstereo_bmcv.py       | LightStereo-S-SceneFlow_fp16_1b_2core.bmodel |    0.456 |
| SE9-16       | lightstereo_bmcv.py       | LightStereo-S-SceneFlow_int8_1b_2core.bmodel |    0.412 |
| SE9-16       | lightstereo_bmcv.py       | LightStereo-S-SceneFlow_int8_4b_2core.bmodel |    0.412 |
| SE9-16       | lightstereo_bmcv.soc      | LightStereo-S-SceneFlow_fp32_1b_2core.bmodel |    0.457 |
| SE9-16       | lightstereo_bmcv.soc      | LightStereo-S-SceneFlow_fp16_1b_2core.bmodel |    0.456 |
| SE9-16       | lightstereo_bmcv.soc      | LightStereo-S-SceneFlow_int8_1b_2core.bmodel |    0.412 |
| SE9-16       | lightstereo_bmcv.soc      | LightStereo-S-SceneFlow_int8_4b_2core.bmodel |    0.412 |
| SE9-8        | lightstereo_opencv.py     | LightStereo-S-SceneFlow_fp32_1b.bmodel |    0.454 |
| SE9-8        | lightstereo_opencv.py     | LightStereo-S-SceneFlow_fp16_1b.bmodel |    0.453 |
| SE9-8        | lightstereo_opencv.py     | LightStereo-S-SceneFlow_int8_1b.bmodel |    0.405 |
| SE9-8        | lightstereo_opencv.py     | LightStereo-S-SceneFlow_int8_4b.bmodel |    0.405 |
| SE9-8        | lightstereo_bmcv.py       | LightStereo-S-SceneFlow_fp32_1b.bmodel |    0.457 |
| SE9-8        | lightstereo_bmcv.py       | LightStereo-S-SceneFlow_fp16_1b.bmodel |    0.456 |
| SE9-8        | lightstereo_bmcv.py       | LightStereo-S-SceneFlow_int8_1b.bmodel |    0.412 |
| SE9-8        | lightstereo_bmcv.py       | LightStereo-S-SceneFlow_int8_4b.bmodel |    0.412 |
| SE9-8        | lightstereo_bmcv.soc      | LightStereo-S-SceneFlow_fp32_1b.bmodel |    0.457 |
| SE9-8        | lightstereo_bmcv.soc      | LightStereo-S-SceneFlow_fp16_1b.bmodel |    0.456 |
| SE9-8        | lightstereo_bmcv.soc      | LightStereo-S-SceneFlow_int8_1b.bmodel |    0.412 |
| SE9-8        | lightstereo_bmcv.soc      | LightStereo-S-SceneFlow_int8_4b.bmodel |    0.412 |

> **测试说明**：  
> 1. 由于sdk版本之间可能存在差异，实际运行结果与本表有<0.01的精度误差是正常的；
> 2. 在搭载了相同TPU和SOPHONSDK的PCIe或SoC平台上，相同程序的精度一致，SE5系列对应BM1684，SE7系列对应BM1684X，SE9系列中，SE9-16对应BM1688，SE9-8对应CV186X；

## 6. 性能测试
### 6.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684X/LightStereo-S-SceneFlow_fp16_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

|                  测试模型                         | calculate time(ms) |
| -------------------------------------------       | ----------------- |
| BM1684X/LightStereo-S-SceneFlow_fp32_1b.bmodel|          86.91  |
| BM1684X/LightStereo-S-SceneFlow_fp16_1b.bmodel|          27.35  |
| BM1684X/LightStereo-S-SceneFlow_int8_1b.bmodel|          21.46  |
| BM1684X/LightStereo-S-SceneFlow_int8_4b.bmodel|          20.49  |
| BM1688/LightStereo-S-SceneFlow_fp32_1b.bmodel|         321.78  |
| BM1688/LightStereo-S-SceneFlow_fp16_1b.bmodel|          75.31  |
| BM1688/LightStereo-S-SceneFlow_int8_1b.bmodel|          40.68  |
| BM1688/LightStereo-S-SceneFlow_int8_4b.bmodel|          36.08  |
| BM1688/LightStereo-S-SceneFlow_fp32_1b_2core.bmodel|         191.59  |
| BM1688/LightStereo-S-SceneFlow_fp16_1b_2core.bmodel|          53.80  |
| BM1688/LightStereo-S-SceneFlow_int8_1b_2core.bmodel|          32.51  |
| BM1688/LightStereo-S-SceneFlow_int8_4b_2core.bmodel|          25.93  |
| CV186X/LightStereo-S-SceneFlow_fp32_1b.bmodel|         354.74  |
| CV186X/LightStereo-S-SceneFlow_fp16_1b.bmodel|          91.46  |
| CV186X/LightStereo-S-SceneFlow_int8_1b.bmodel|          50.23  |
| CV186X/LightStereo-S-SceneFlow_int8_4b.bmodel|          47.16  |

> **测试说明**：  
> 1. 性能测试结果具有一定的波动性；
> 2. `calculate time`已折算为平均每张图片的推理时间；
> 3. SoC和PCIe的测试结果基本一致。

### 6.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++和Python例程打印的时间已经折算为单张图片的处理时间。在不同的测试平台上，使用不同的例程、模型测试`datasets/KITTI12/kitti12_train194.txt`，性能测试结果如下：
|    测试平台  |     测试程序      |             测试模型                |decode_time    |preprocess_time  |inference_time   |postprocess_time| 
| ----------- | ---------------- | ----------------------------------- | --------      | ---------       | ---------        | --------- |
|   SE7-32    |lightstereo_opencv.py|LightStereo-S-SceneFlow_fp32_1b.bmodel|      42.84      |     154.08      |     147.39      |      0.03       |
|   SE7-32    |lightstereo_opencv.py|LightStereo-S-SceneFlow_fp16_1b.bmodel|      42.82      |     151.30      |      92.38      |      0.03       |
|   SE7-32    |lightstereo_opencv.py|LightStereo-S-SceneFlow_int8_1b.bmodel|      42.82      |     152.70      |     111.64      |      0.03       |
|   SE7-32    |lightstereo_opencv.py|LightStereo-S-SceneFlow_int8_4b.bmodel|      43.14      |     164.69      |     102.68      |      0.01       |
|   SE7-32    |lightstereo_bmcv.py|LightStereo-S-SceneFlow_fp32_1b.bmodel|      52.41      |      6.95       |      90.87      |      0.12       |
|   SE7-32    |lightstereo_bmcv.py|LightStereo-S-SceneFlow_fp16_1b.bmodel|      52.37      |      6.94       |      31.23      |      0.12       |
|   SE7-32    |lightstereo_bmcv.py|LightStereo-S-SceneFlow_int8_1b.bmodel|      52.46      |      3.99       |      25.29      |      0.12       |
|   SE7-32    |lightstereo_bmcv.py|LightStereo-S-SceneFlow_int8_4b.bmodel|      52.11      |      4.43       |      23.75      |      0.05       |
|   SE7-32    |lightstereo_bmcv.soc|LightStereo-S-SceneFlow_fp32_1b.bmodel|      50.02      |      3.69       |      87.14      |      1.86       |
|   SE7-32    |lightstereo_bmcv.soc|LightStereo-S-SceneFlow_fp16_1b.bmodel|      50.02      |      3.68       |      27.53      |      1.86       |
|   SE7-32    |lightstereo_bmcv.soc|LightStereo-S-SceneFlow_int8_1b.bmodel|      50.05      |      2.26       |      21.64      |      1.86       |
|   SE7-32    |lightstereo_bmcv.soc|LightStereo-S-SceneFlow_int8_4b.bmodel|      49.93      |      2.15       |      21.10      |      1.89       |
|   SE9-16    |lightstereo_opencv.py|LightStereo-S-SceneFlow_fp32_1b.bmodel|      59.51      |     173.58      |     382.81      |      0.05       |
|   SE9-16    |lightstereo_opencv.py|LightStereo-S-SceneFlow_fp16_1b.bmodel|      59.51      |     172.39      |     135.88      |      0.05       |
|   SE9-16    |lightstereo_opencv.py|LightStereo-S-SceneFlow_int8_1b.bmodel|      59.54      |     172.94      |     142.96      |      0.05       |
|   SE9-16    |lightstereo_opencv.py|LightStereo-S-SceneFlow_int8_4b.bmodel|      59.39      |     169.29      |     133.28      |      0.02       |
|   SE9-16    |lightstereo_bmcv.py|LightStereo-S-SceneFlow_fp32_1b.bmodel|      73.24      |      11.92      |     326.73      |      0.16       |
|   SE9-16    |lightstereo_bmcv.py|LightStereo-S-SceneFlow_fp16_1b.bmodel|      73.13      |      11.87      |      80.14      |      0.16       |
|   SE9-16    |lightstereo_bmcv.py|LightStereo-S-SceneFlow_int8_1b.bmodel|      73.22      |      8.47       |      45.39      |      0.16       |
|   SE9-16    |lightstereo_bmcv.py|LightStereo-S-SceneFlow_int8_4b.bmodel|      72.83      |      9.27       |      40.58      |      0.06       |
|   SE9-16    |lightstereo_bmcv.soc|LightStereo-S-SceneFlow_fp32_1b.bmodel|      70.37      |      7.40       |     322.01      |      2.65       |
|   SE9-16    |lightstereo_bmcv.soc|LightStereo-S-SceneFlow_fp16_1b.bmodel|      70.30      |      7.38       |      75.54      |      2.60       |
|   SE9-16    |lightstereo_bmcv.soc|LightStereo-S-SceneFlow_int8_1b.bmodel|      70.27      |      6.18       |      40.84      |      2.61       |
|   SE9-16    |lightstereo_bmcv.soc|LightStereo-S-SceneFlow_int8_4b.bmodel|      70.19      |      6.01       |      37.12      |      2.64       |
|   SE9-16    |lightstereo_opencv.py|LightStereo-S-SceneFlow_fp32_1b_2core.bmodel|      59.60      |     172.34      |     251.55      |      0.05       |
|   SE9-16    |lightstereo_opencv.py|LightStereo-S-SceneFlow_fp16_1b_2core.bmodel|      59.50      |     172.05      |     114.37      |      0.05       |
|   SE9-16    |lightstereo_opencv.py|LightStereo-S-SceneFlow_int8_1b_2core.bmodel|      59.46      |     172.54      |     134.06      |      0.05       |
|   SE9-16    |lightstereo_opencv.py|LightStereo-S-SceneFlow_int8_4b_2core.bmodel|      59.43      |     169.30      |     121.84      |      0.02       |
|   SE9-16    |lightstereo_bmcv.py|LightStereo-S-SceneFlow_fp32_1b_2core.bmodel|      73.11      |      11.92      |     196.12      |      0.16       |
|   SE9-16    |lightstereo_bmcv.py|LightStereo-S-SceneFlow_fp16_1b_2core.bmodel|      73.15      |      11.88      |      58.55      |      0.16       |
|   SE9-16    |lightstereo_bmcv.py|LightStereo-S-SceneFlow_int8_1b_2core.bmodel|      73.05      |      8.44       |      37.37      |      0.16       |
|   SE9-16    |lightstereo_bmcv.py|LightStereo-S-SceneFlow_int8_4b_2core.bmodel|      72.88      |      9.37       |      30.24      |      0.06       |
|   SE9-16    |lightstereo_bmcv.soc|LightStereo-S-SceneFlow_fp32_1b_2core.bmodel|      70.32      |      7.41       |     191.49      |      2.60       |
|   SE9-16    |lightstereo_bmcv.soc|LightStereo-S-SceneFlow_fp16_1b_2core.bmodel|      70.24      |      7.44       |      54.02      |      2.60       |
|   SE9-16    |lightstereo_bmcv.soc|LightStereo-S-SceneFlow_int8_1b_2core.bmodel|      70.24      |      6.18       |      32.71      |      2.60       |
|   SE9-16    |lightstereo_bmcv.soc|LightStereo-S-SceneFlow_int8_4b_2core.bmodel|      70.20      |      6.04       |      26.75      |      2.64       |
|    SE9-8    |lightstereo_opencv.py|LightStereo-S-SceneFlow_fp32_1b.bmodel|      73.49      |     174.39      |     415.77      |      0.05       |
|    SE9-8    |lightstereo_opencv.py|LightStereo-S-SceneFlow_fp16_1b.bmodel|      59.68      |     171.47      |     151.25      |      0.05       |
|    SE9-8    |lightstereo_opencv.py|LightStereo-S-SceneFlow_int8_1b.bmodel|      59.31      |     173.13      |     152.51      |      0.05       |
|    SE9-8    |lightstereo_opencv.py|LightStereo-S-SceneFlow_int8_4b.bmodel|      66.90      |     181.73      |     148.30      |      0.02       |
|    SE9-8    |lightstereo_bmcv.py|LightStereo-S-SceneFlow_fp32_1b.bmodel|      77.28      |      11.85      |     359.70      |      0.16       |
|    SE9-8    |lightstereo_bmcv.py|LightStereo-S-SceneFlow_fp16_1b.bmodel|      73.08      |      11.85      |      96.23      |      0.15       |
|    SE9-8    |lightstereo_bmcv.py|LightStereo-S-SceneFlow_int8_1b.bmodel|      72.91      |      8.43       |      54.99      |      0.16       |
|    SE9-8    |lightstereo_bmcv.py|LightStereo-S-SceneFlow_int8_4b.bmodel|      72.68      |      9.41       |      51.68      |      0.06       |
|    SE9-8    |lightstereo_bmcv.soc|LightStereo-S-SceneFlow_fp32_1b.bmodel|      70.21      |      7.42       |     355.02      |      2.64       |
|    SE9-8    |lightstereo_bmcv.soc|LightStereo-S-SceneFlow_fp16_1b.bmodel|      70.17      |      7.42       |      91.63      |      2.60       |
|    SE9-8    |lightstereo_bmcv.soc|LightStereo-S-SceneFlow_int8_1b.bmodel|      70.29      |      6.18       |      50.43      |      2.61       |
|    SE9-8    |lightstereo_bmcv.soc|LightStereo-S-SceneFlow_int8_4b.bmodel|      70.14      |      6.00       |      48.39      |      2.67       |

> **测试说明**：  
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. SE5-16/SE7-32的主控处理器均为8核CA53@2.3GHz，SE9-16为8核CA53@1.6GHz，SE9-8为6核CA53@1.6GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 4. 图片分辨率对解码时间影响较大，不同的测试图片可能存在较大差异。 

## 7. FAQ
其他问题请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。