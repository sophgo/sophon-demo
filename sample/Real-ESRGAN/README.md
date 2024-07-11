# Real-ESRGAN

## 目录

* [1. 简介](#1-简介)
* [2. 特性](#2-特性)
* [3. 准备模型与数据](#3-准备模型与数据)
* [4. 模型编译](#4-模型编译)
* [5. 例程测试](#5-例程测试)
* [6. 精度测试](#6-精度测试)
* [7. 性能测试](#7-性能测试)
  * [7.1 bmrt_test](#71-bmrt_test)
  * [7.2 程序运行性能](#72-程序运行性能)
* [8. FAQ](#8-faq)
  
## 1. 简介
本例程对[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)的`realesr-general-x4v3`轻量级超分模型进行移植，使之能在SOPHON BM1684X/BM1688/CV186X 上进行推理测试。

## 2. 特性
* 支持BM1688(SoC)、BM1684X(x86 PCIe、SoC)、BM1684(x86 PCIe、SoC、arm PCIe)
* 支持FP32、FP16(BM1684X/BM1688)、INT8模型编译和推理
* 支持基于OpenCV和BMCV预处理的Python推理
* 支持单batch和多batch模型推理
* 支持图片测试
 
## 3. 准备模型与数据
建议使用TPU-MLIR编译BModel，Pytorch模型在编译前要导出成onnx模型，如果您使用的tpu-mlir版本>=v1.3.0（即官网v23.07.01），可以直接使用torchscript模型。具体可参考[Real-ESRGAN模型导出](./docs/Real-ESRGAN_Export_Guide.md)。

​同时，您需要准备用于测试的数据集，如果量化模型，还要准备用于量化的数据集。

​本例程在`scripts`目录下提供了相关模型和数据集的下载脚本`download.sh`，您也可以自己准备模型和数据集，并参考[4. 模型编译](#4-模型编译)进行模型转换。

```bash
# 安装unzip，若已安装请跳过，非ubuntu系统视情况使用yum或其他方式安装
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
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
│   ├── real_esrgan_fp32_1b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1，num_core=2
│   ├── real_esrgan_fp16_1b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1，num_core=2
│   ├── real_esrgan_int8_1b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=1，num_core=2
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
| BM1684X/real_esrgan_fp32_1b.bmodel     |         711.05     |
| BM1684X/real_esrgan_fp16_1b.bmodel     |          64.20     |
| BM1684X/real_esrgan_int8_1b.bmodel     |          32.57     |
| BM1684X/real_esrgan_int8_4b.bmodel     |          31.69     |
| BM1688/real_esrgan_fp32_1b.bmodel      |        3754.10     |
| BM1688/real_esrgan_fp16_1b.bmodel      |         455.90     |
| BM1688/real_esrgan_int8_1b.bmodel      |         122.50     |
| BM1688/real_esrgan_int8_4b.bmodel      |         120.50     |
| BM1688/real_esrgan_fp32_1b_2core.bmodel|        1909.69     |
| BM1688/real_esrgan_fp16_1b_2core.bmodel|         247.33     |
| BM1688/real_esrgan_int8_1b_2core.bmodel|          94.02     |
| BM1688/real_esrgan_int8_4b_2core.bmodel|          66.15     |
| CV186X/real_esrgan_fp32_1b.bmodel      |        3741.93     |
| CV186X/real_esrgan_fp16_1b.bmodel      |         451.70     |
| CV186X/real_esrgan_int8_1b.bmodel      |         120.87     |
| CV186X/real_esrgan_int8_4b.bmodel      |         118.22     |

> **测试说明**：  
> 1. 性能测试结果具有一定的波动性；
> 2. `calculate time`已折算为平均每张图片的推理时间；
> 3. SoC和PCIe的测试结果基本一致。

### 7.2 程序运行性能
参考[C++例程](cpp/README.md#3-推理测试)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++和Python例程打印的时间已经折算为单张图片的处理时间。

在不同的测试平台上，使用不同的例程、模型测试，性能测试结果如下：
|   测试平台  |     测试程序        |             测试模型               |   decode_time   | preprocess_time | inference_time  |postprocess_time  |
| ----------  | -----------------   | -----------------------------------|-----------------|-----------------|-----------------|-----------------|
|   SE7-32    |real_esrgan_opencv.py|    real_esrgan_fp32_1b.bmodel      |      1.86       |      18.56      |     761.65      |      80.92      |
|   SE7-32    |real_esrgan_opencv.py|    real_esrgan_fp16_1b.bmodel      |      1.84       |      18.70      |     114.75      |      72.18      |
|   SE7-32    |real_esrgan_opencv.py|    real_esrgan_int8_1b.bmodel      |      1.83       |      18.46      |     344.08      |      71.63      |
|   SE7-32    |real_esrgan_opencv.py|    real_esrgan_int8_4b.bmodel      |      1.76       |      19.39      |     342.58      |      83.54      |
|   SE7-32    |real_esrgan_bmcv.py|    real_esrgan_fp32_1b.bmodel        |      1.75       |      1.92       |     722.50      |     106.98      |
|   SE7-32    |real_esrgan_bmcv.py|    real_esrgan_fp16_1b.bmodel        |      1.73       |      1.93       |      75.53      |     106.83      |
|   SE7-32    |real_esrgan_bmcv.py|    real_esrgan_int8_1b.bmodel        |      1.75       |      1.51       |      35.57      |      58.49      |
|   SE7-32    |real_esrgan_bmcv.py|    real_esrgan_int8_4b.bmodel        |      1.41       |      1.36       |      34.53      |      58.79      |
|   SE7-32    |real_esrgan_bmcv.soc|    real_esrgan_fp32_1b.bmodel       |      2.25       |      0.61       |     711.02      |      51.60      |
|   SE7-32    |real_esrgan_bmcv.soc|    real_esrgan_fp16_1b.bmodel       |      2.25       |      0.61       |      64.18      |      51.72      |
|   SE7-32    |real_esrgan_bmcv.soc|    real_esrgan_int8_1b.bmodel       |      2.27       |      0.46       |      32.59      |      93.60      |
|   SE7-32    |real_esrgan_bmcv.soc|    real_esrgan_int8_4b.bmodel       |      1.95       |      0.43       |      31.69      |      91.32      |
|    SE9-8    |real_esrgan_opencv.py|    real_esrgan_fp32_1b.bmodel      |      13.90      |      43.32      |     3803.27     |      90.92      |
|    SE9-8    |real_esrgan_opencv.py|    real_esrgan_fp16_1b.bmodel      |      3.21       |      42.89      |     512.60      |      87.89      |
|    SE9-8    |real_esrgan_opencv.py|    real_esrgan_int8_1b.bmodel      |      3.20       |      42.98      |     537.23      |      87.68      |
|    SE9-8    |real_esrgan_opencv.py|    real_esrgan_int8_4b.bmodel      |      8.46       |      47.41      |     541.77      |     213.41      |
|    SE9-8    |real_esrgan_bmcv.py|    real_esrgan_fp32_1b.bmodel        |      13.33      |      3.73       |     3758.24     |     131.44      |
|    SE9-8    |real_esrgan_bmcv.py|    real_esrgan_fp16_1b.bmodel        |      3.22       |      3.73       |     467.73      |     132.85      |
|    SE9-8    |real_esrgan_bmcv.py|    real_esrgan_int8_1b.bmodel        |      3.21       |      3.38       |     125.12      |      74.66      |
|    SE9-8    |real_esrgan_bmcv.py|    real_esrgan_int8_4b.bmodel        |      2.86       |      3.02       |     122.24      |     106.95      |
|    SE9-8    |real_esrgan_bmcv.soc|    real_esrgan_fp32_1b.bmodel       |      5.54       |      1.60       |     3741.92     |     108.41      |
|    SE9-8    |real_esrgan_bmcv.soc|    real_esrgan_fp16_1b.bmodel       |      4.82       |      1.59       |     451.58      |     108.37      |
|    SE9-8    |real_esrgan_bmcv.soc|    real_esrgan_int8_1b.bmodel       |      5.02       |      1.58       |     120.78      |     121.39      |
|    SE9-8    |real_esrgan_bmcv.soc|    real_esrgan_int8_4b.bmodel       |      4.53       |      1.49       |     118.20      |     121.00      |
|   SE9-16    |real_esrgan_opencv.py|    real_esrgan_fp32_1b.bmodel      |      3.32       |      24.22      |     3791.12     |      88.00      |
|   SE9-16    |real_esrgan_opencv.py|    real_esrgan_fp16_1b.bmodel      |      3.26       |      23.95      |     500.79      |      88.22      |
|   SE9-16    |real_esrgan_opencv.py|    real_esrgan_int8_1b.bmodel      |      3.26       |      23.90      |     548.49      |      88.13      |
|   SE9-16    |real_esrgan_opencv.py|    real_esrgan_int8_4b.bmodel      |      3.14       |      24.95      |     545.45      |     113.01      |
|   SE9-16    |real_esrgan_bmcv.py|    real_esrgan_fp32_1b.bmodel        |      3.24       |      3.81       |     3746.02     |     132.69      |
|   SE9-16    |real_esrgan_bmcv.py|    real_esrgan_fp16_1b.bmodel        |      3.26       |      3.85       |     455.66      |     132.52      |
|   SE9-16    |real_esrgan_bmcv.py|    real_esrgan_int8_1b.bmodel        |      3.24       |      3.46       |     120.55      |      76.49      |
|   SE9-16    |real_esrgan_bmcv.py|    real_esrgan_int8_4b.bmodel        |      2.93       |      3.08       |     118.63      |      76.70      |
|   SE9-16    |real_esrgan_bmcv.soc|    real_esrgan_fp32_1b.bmodel       |      3.64       |      1.52       |     3729.57     |      65.26      |
|   SE9-16    |real_esrgan_bmcv.soc|    real_esrgan_fp16_1b.bmodel       |      3.64       |      1.53       |     439.27      |      65.06      |
|   SE9-16    |real_esrgan_bmcv.soc|    real_esrgan_int8_1b.bmodel       |      3.66       |      1.52       |     116.17      |      80.13      |
|   SE9-16    |real_esrgan_bmcv.soc|    real_esrgan_int8_4b.bmodel       |      3.40       |      1.43       |     114.54      |      77.70      |
|   SE9-16    |real_esrgan_opencv.py| real_esrgan_fp32_1b_2core.bmodel   |      3.30       |      24.39      |     1946.97     |      88.10      |
|   SE9-16    |real_esrgan_opencv.py| real_esrgan_fp16_1b_2core.bmodel   |      3.28       |      24.25      |     292.25      |      88.05      |
|   SE9-16    |real_esrgan_opencv.py| real_esrgan_int8_1b_2core.bmodel   |      3.30       |      24.10      |     520.30      |      88.24      |
|   SE9-16    |real_esrgan_opencv.py| real_esrgan_int8_4b_2core.bmodel   |      3.17       |      24.98      |     490.98      |     106.71      |
|   SE9-16    |real_esrgan_bmcv.py| real_esrgan_fp32_1b_2core.bmodel     |      3.24       |      3.87       |     1901.67     |     131.68      |
|   SE9-16    |real_esrgan_bmcv.py| real_esrgan_fp16_1b_2core.bmodel     |      3.23       |      3.85       |     247.06      |     133.33      |
|   SE9-16    |real_esrgan_bmcv.py| real_esrgan_int8_1b_2core.bmodel     |      3.25       |      3.48       |      92.04      |      76.11      |
|   SE9-16    |real_esrgan_bmcv.py| real_esrgan_int8_4b_2core.bmodel     |      2.96       |      3.07       |      64.27      |      76.60      |
|   SE9-16    |real_esrgan_bmcv.soc| real_esrgan_fp32_1b_2core.bmodel    |      3.65       |      1.52       |     1885.24     |      64.88      |
|   SE9-16    |real_esrgan_bmcv.soc| real_esrgan_fp16_1b_2core.bmodel    |      3.65       |      1.52       |     230.78      |      64.88      |
|   SE9-16    |real_esrgan_bmcv.soc| real_esrgan_int8_1b_2core.bmodel    |      3.64       |      1.51       |      87.69      |      77.74      |
|   SE9-16    |real_esrgan_bmcv.soc| real_esrgan_int8_4b_2core.bmodel    |      3.44       |      1.43       |      60.18      |      77.54      |


> **测试说明**：  
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. SE5-16/SE7-16的主控处理器均为8核 ARM A53 42320 DMIPS @2.3GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异，不同的阈值对后处理时间影响较大。 

## 8. FAQ
请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。