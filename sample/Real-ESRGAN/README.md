# Real-ESRGAN

## 目录

- [Real-ESRGAN](#real-esrgan)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
    - [2.1 目录结构说明](#21-目录结构说明)
    - [2.2 SDK特性](#22-sdk特性)
  - [3. 准备模型与数据](#3-准备模型与数据)
  - [4. 模型编译](#4-模型编译)
  - [5. 例程测试](#5-例程测试)
  - [6. 精度测试](#6-精度测试)
  - [7. 性能测试](#7-性能测试)
    - [7.1 bmrt\_test](#71-bmrt_test)
    - [7.2 程序运行性能](#72-程序运行性能)
  - [8. FAQ](#8-faq)
  
## 1. 简介
本例程对[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)的`realesr-general-x4v3`轻量级超分模型进行移植，使之能在SOPHON BM1684X/BM1688/CV186X 上进行推理测试。

## 2. 特性

### 2.1 目录结构说明
```bash
├── cpp                   # 存放C++例程及其README
|   ├──README.md      
|   ├──real_esrgan_bmcv        # C++例程
├── docs                  # 存放本例程专用文档，如ONNX导出、移植常见问题等
├── pics                  # 存放README等说明文档中用到的图片
├── python                # 存放Python例程及其README
|   ├──README.md 
|   ├──real_esrgan_bmcv.py   # 使用bmcv做预处理、bmodel推理的Python例程
|   ├──real_esrgan_opencv.py # 使用opencv做预处理、bmodel推理的Python例程
|   ├──real_esrgan_onnx.py   # 使用opencv预处理、onnx推理的Python例程
├── README.md             # 本例程的中文指南
├── scripts               # 存放模型编译、数据下载、自动测试等shell脚本
└── tools                 # 存放精度测试、性能比对等python脚本
```

### 2.2 SDK特性
* 支持BM1688(SoC)和BM1684X(x86 PCIe、SoC、riscv PCIe)
* 支持FP32、FP16(BM1684X/BM1688)、INT8模型编译和推理
* 支持C++、Python推理
* 支持图片测试
 
## 3. 准备模型与数据
建议使用TPU-MLIR编译BModel，Pytorch模型在编译前要导出成onnx模型，如果您使用的tpu-mlir版本>=v1.3.0（即官网v23.07.01），可以直接使用torchscript模型。具体可参考[Real-ESRGAN模型导出](./docs/Real-ESRGAN_Export_Guide.md)。

​同时，您需要准备用于测试的数据集，如果量化模型，还要准备用于量化的数据集。

​本例程在`scripts`目录下提供了相关模型和数据集的下载脚本`download.sh`，您也可以自己准备模型和数据集，并参考[4. 模型编译](#4-模型编译)进行模型转换。

```bash
chmod -R +x scripts/
./scripts/download.sh --all 
```

`download.sh`默认只下载`datasets`，`models`可以通过指定参数分平台下载，参数如下：
```bash
--all     # 下载所有模型
--BM1684X # 下载BM1684X的bmodel
--BM1688  # 下载BM1688的bmodel
--CV186X  # 下载BM1688的bmodel
--onnx    # 下载onnx
```

下载的模型包括：
```
./models
├── BM1684X    
│   ├── real_esrgan_fp32_1b.bmodel       # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=1
│   ├── real_esrgan_fp16_1b.bmodel       # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=1
│   ├── real_esrgan_int8_1b.bmodel       # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=1
│   └── real_esrgan_int8_4b.bmodel       # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=4
├── BM1688
|   ├── real_esrgan_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1，num_core=1
│   ├── real_esrgan_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1，num_core=1
│   ├── real_esrgan_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=1，num_core=1
│   ├── real_esrgan_int8_4b.bmodel   # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=4，num_core=1
│   └── real_esrgan_int8_4b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=4，num_core=2
├── CV186X
│   ├── real_esrgan_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于CV186X的FP32 BModel，batch_size=1
│   ├── real_esrgan_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于CV186X的FP16 BModel，batch_size=1
│   ├── real_esrgan_int8_1b.bmodel   # 使用TPU-MLIR编译，用于CV186X的INT8 BModel，batch_size=1
│   └── real_esrgan_int8_4b.bmodel   # 使用TPU-MLIR编译，用于CV186X的INT8 BModel，batch_size=4
└── onnx
    └── realesr-general-x4v3t.onnx             # 导出的onnx动态模型       
```
下载的数据包括：
```
./datasets                                     
├── coco128                                   # coco128数据集，测试图片      
```
## 4. 模型编译
参考[onnx导出指南](docs/export_onnx_guide.md)来导出onnx，导出的onnx还需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。建议使用TPU-MLIR编译BModel。

模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index/material/all/all.html)相应版本的SDK中获取)。

- 生成FP32 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684x #bm1684x/bm1688
```

​执行上述命令会在`models/bm1684x`等文件夹下生成`real_esrgan_fp32_1b.bmodel`文件，即转换好的FP32 BModel。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x #bm1688
```

​执行上述命令会在`models/BM1684X/`等文件夹下生成`real_esrgan_fp16_1b.bmodel`文件，即转换好的FP16 BModel。

- 生成INT8 BModel

​本例程在`scripts`目录下提供了量化INT8 BModel的脚本，请注意修改`gen_int8bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台（**BM1684X/BM1688/CV186X**），如：

```shell
./scripts/gen_int8bmodel_mlir.sh bm1684x #bm1684x/bm1688
```

​上述脚本会在`models/bm1684x`等文件夹下生成`real_esrgan_int8_1b.bmodel`等文件，即转换好的INT8 BModel。
## 5. 例程测试
- [Python例程](./python/README.md)

## 6. 精度测试
暂不提供精度测试结果。
## 7. 性能测试
### 7.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684X/
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

| 测试模型                               | calculate time(ms) |
| ---------------------------------------| -------------------|
| BM1684X/real_esrgan_fp32_1b.bmodel |         711.62  |
| BM1684X/real_esrgan_fp16_1b.bmodel |          63.91  |
| BM1684X/real_esrgan_int8_1b.bmodel |          32.52  |
| BM1684X/real_esrgan_int8_4b.bmodel |          31.76  |
| BM1688/real_esrgan_fp32_1b.bmodel  |        3729.99  |
| BM1688/real_esrgan_fp16_1b.bmodel  |         438.48  |
| BM1688/real_esrgan_int8_1b.bmodel  |         115.69  |
| BM1688/real_esrgan_int8_4b.bmodel  |         114.72  |
| BM1688/real_esrgan_int8_4b_2core.bmodel|          64.63  |

> **测试说明**：  
> 1. 性能测试结果具有一定的波动性；
> 2. `calculate time`已折算为平均每张图片的推理时间；
> 3. SoC和PCIe的测试结果基本一致。

### 7.2 程序运行性能
参考[C++例程](cpp/README.md#3-推理测试)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++和Python例程打印的时间已经折算为单张图片的处理时间。

在不同的测试平台上，使用不同的例程、模型测试，性能测试结果如下：
|   测试平台  |     测试程序        |             测试模型               |   decode_time   | preprocess_time | inference_time  |postprocess_time  |
| ----------  | -----------------   | -----------------------------------|-----------------|-----------------|-----------------|-----------------|
|   SE7-32    |real_esrgan_opencv.py|    real_esrgan_fp32_1b.bmodel     |      1.87       |      18.88      |     766.08   |      71.21      |
|   SE7-32    |real_esrgan_opencv.py|    real_esrgan_fp16_1b.bmodel     |      1.86       |      18.61      |     114.84   |      71.46      |
|   SE7-32    |real_esrgan_opencv.py|    real_esrgan_int8_1b.bmodel     |      1.85       |      18.60      |     344.06   |      71.41      |
|   SE7-32    |real_esrgan_opencv.py|    real_esrgan_int8_4b.bmodel     |      1.78       |      19.32      |     342.83    |      79.77      |
|   SE7-32    |real_esrgan_bmcv.py|    real_esrgan_fp32_1b.bmodel     |      1.81       |      1.96       |     723.02      |  109.33      |
|   SE7-32    |real_esrgan_bmcv.py|    real_esrgan_fp16_1b.bmodel     |      1.81       |      1.97       |      75.08      |  109.82      |
|   SE7-32    |real_esrgan_bmcv.py|    real_esrgan_int8_1b.bmodel     |      1.83       |      1.55       |      35.46      |   58.71      |
|   SE7-32    |real_esrgan_bmcv.py|    real_esrgan_int8_4b.bmodel     |      1.46       |      1.38       |      34.57      |   60.03      |
|   SE7-32    |real_esrgan_bmcv.soc|    real_esrgan_fp32_1b.bmodel     |      1.32       |      1.11       |     724.83      |   97.69      |
|   SE7-32    |real_esrgan_bmcv.soc|    real_esrgan_fp16_1b.bmodel     |      1.34       |      1.11       |      76.64      |   97.63      |
|   SE7-32    |real_esrgan_bmcv.soc|    real_esrgan_int8_1b.bmodel     |      1.30       |      0.69       |      33.92      |   3.10       |
|   SE7-32    |real_esrgan_bmcv.soc|    real_esrgan_int8_4b.bmodel     |      1.14       |      0.63       |      33.13      |   3.09       |
|   SE9-16    |real_esrgan_opencv.py|    real_esrgan_fp32_1b.bmodel     |      4.09       |      24.75      |     3794.30     |      89.71      |
|   SE9-16    |real_esrgan_opencv.py|    real_esrgan_fp16_1b.bmodel     |      3.38       |      24.39      |     503.90      |      89.40      |
|   SE9-16    |real_esrgan_opencv.py|    real_esrgan_int8_1b.bmodel     |      3.35       |      24.42      |     549.36      |      90.44      |
|   SE9-16    |real_esrgan_opencv.py|    real_esrgan_int8_4b.bmodel     |      3.11       |      24.95      |     546.77      |      88.84      |
|   SE9-16    |real_esrgan_opencv.py| real_esrgan_int8_4b_2core.bmodel  |      3.12       |      24.91      |     142.10      |      89.03      |
|   SE9-16    |real_esrgan_bmcv.py|    real_esrgan_fp32_1b.bmodel     |      3.23       |      3.59       |     3746.51     |     138.87      |
|   SE9-16    |real_esrgan_bmcv.py|    real_esrgan_fp16_1b.bmodel     |      3.28       |      3.63       |     455.04      |     139.14      |
|   SE9-16    |real_esrgan_bmcv.py|    real_esrgan_int8_1b.bmodel     |      3.25       |      3.24       |     119.96      |      75.59      |
|   SE9-16    |real_esrgan_bmcv.py|    real_esrgan_int8_4b.bmodel     |      2.85       |      2.87       |     118.97      |      76.46      |
|   SE9-16    |real_esrgan_bmcv.py| real_esrgan_int8_4b_2core.bmodel  |      2.85       |      2.86       |      64.48      |      75.93      |
|   SE9-16    |real_esrgan_bmcv.soc|    real_esrgan_fp32_1b.bmodel     |      2.49       |      2.14       |     3738.22     |     130.09      |
|   SE9-16    |real_esrgan_bmcv.soc|    real_esrgan_fp16_1b.bmodel     |      2.50       |      2.13       |     446.65      |     129.98      |
|   SE9-16    |real_esrgan_bmcv.soc|    real_esrgan_int8_1b.bmodel     |      2.43       |      1.74       |     117.58      |      10.39      |
|   SE9-16    |real_esrgan_bmcv.soc|    real_esrgan_int8_4b.bmodel     |      2.18       |      1.61       |     116.93      |      10.40      |
|   SE9-16    |real_esrgan_bmcv.soc| real_esrgan_int8_4b_2core.bmodel  |      2.18       |      1.61       |      62.47      |      10.37      |
|    SE9-8    |real_esrgan_opencv.py|    real_esrgan_fp32_1b.bmodel     |      17.15      |      24.43      |     3813.90     |     132.27      |
|    SE9-8    |real_esrgan_opencv.py|    real_esrgan_fp16_1b.bmodel     |      5.63       |      24.49      |     516.81      |     132.07      |
|    SE9-8    |real_esrgan_opencv.py|    real_esrgan_int8_1b.bmodel     |      5.66       |      24.60      |     550.77      |     132.19      |
|    SE9-8    |real_esrgan_opencv.py|    real_esrgan_int8_4b.bmodel     |      11.43      |      24.77      |     549.46      |     237.89      |
|    SE9-8    |real_esrgan_bmcv.py|    real_esrgan_fp32_1b.bmodel     |      12.47      |      3.60       |     3765.67     |     142.59      |
|    SE9-8    |real_esrgan_bmcv.py|    real_esrgan_fp16_1b.bmodel     |      3.41       |      3.57       |     466.11      |     142.27      |
|    SE9-8    |real_esrgan_bmcv.py|    real_esrgan_int8_1b.bmodel     |      3.46       |      3.21       |     123.81      |      77.01      |
|    SE9-8    |real_esrgan_bmcv.py|    real_esrgan_int8_4b.bmodel     |      9.51       |      2.85       |     122.29      |     157.92      |
|    SE9-8    |real_esrgan_bmcv.soc|    real_esrgan_fp32_1b.bmodel     |      16.42      |      2.13       |     3757.22     |     164.12      |
|    SE9-8    |real_esrgan_bmcv.soc|    real_esrgan_fp16_1b.bmodel     |      2.48       |      2.13       |     457.72      |     163.80      |
|    SE9-8    |real_esrgan_bmcv.soc|    real_esrgan_int8_1b.bmodel     |      2.46       |      1.74       |     121.44      |      10.39      |
|    SE9-8    |real_esrgan_bmcv.soc|    real_esrgan_int8_4b.bmodel     |      2.19       |      1.61       |     120.20      |      10.35      |

> **测试说明**：  
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. SE5-16/SE7-16的主控处理器均为8核 ARM A53 42320 DMIPS @2.3GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异，不同的阈值对后处理时间影响较大。 

## 8. FAQ
请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。