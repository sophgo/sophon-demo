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
│   ├── real_esrgan_fp16_4b.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=4，num_core=1
│   ├── real_esrgan_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=1，num_core=1
│   ├── real_esrgan_int8_4b.bmodel   # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=4，num_core=1
│   ├── real_esrgan_fp32_1b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1，num_core=2
│   ├── real_esrgan_fp32_4b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=4，num_core=2
│   ├── real_esrgan_fp16_1b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1，num_core=2
│   ├── real_esrgan_fp16_4b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=4，num_core=2
│   ├── real_esrgan_int8_1b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=1，num_core=2
│   └── real_esrgan_int8_4b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=4，num_core=2
├── CV186X
│   ├── real_esrgan_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于CV186X的FP32 BModel，batch_size=1
│   ├── real_esrgan_fp32_4b.bmodel   # 使用TPU-MLIR编译，用于CV186X的FP32 BModel，batch_size=4
│   ├── real_esrgan_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于CV186X的FP16 BModel，batch_size=1
│   ├── real_esrgan_fp16_4b.bmodel   # 使用TPU-MLIR编译，用于CV186X的FP16 BModel，batch_size=4
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

模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index/material/31/all.html)相应版本的SDK中获取)。

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
|    测试平台  |     测试程序        |             测试模型               |   decode_time   | preprocess_time | inference_time  |postprocess_time |
| ----------  | -----------------    ------------------------------------|-----------------|-----------------|-----------------|-----------------|
|   SE7-32    |real_esrgan_opencv.py|    real_esrgan_fp32_1b.bmodel      |      10.07      |      17.70      |     761.75      |      71.56      |
|   SE7-32    |real_esrgan_opencv.py|    real_esrgan_fp16_1b.bmodel      |      10.05      |      18.25      |     115.09      |      71.27      |
|   SE7-32    |real_esrgan_opencv.py|    real_esrgan_int8_1b.bmodel      |      10.04      |      17.91      |     332.37      |      71.17      |
|   SE7-32    |real_esrgan_opencv.py|    real_esrgan_int8_4b.bmodel      |      10.12      |      18.15      |     331.62      |      74.14      |
|   SE7-32    |real_esrgan_bmcv.py|    real_esrgan_fp32_1b.bmodel        |      2.16       |      1.98       |     722.67      |     108.27      |
|   SE7-32    |real_esrgan_bmcv.py|    real_esrgan_fp16_1b.bmodel        |      1.85       |      2.00       |      75.89      |     108.32      |
|   SE7-32    |real_esrgan_bmcv.py|    real_esrgan_int8_1b.bmodel        |      1.84       |      1.55       |      35.60      |      58.42      |
|   SE7-32    |real_esrgan_bmcv.py|    real_esrgan_int8_4b.bmodel        |      1.49       |      1.37       |      34.65      |      58.64      |
|   SE7-32    |real_esrgan_bmcv.soc|    real_esrgan_fp32_1b.bmodel       |      1.26       |      0.61       |     711.07      |      51.92      |
|   SE7-32    |real_esrgan_bmcv.soc|    real_esrgan_fp16_1b.bmodel       |      1.26       |      0.61       |      64.19      |      52.00      |
|   SE7-32    |real_esrgan_bmcv.soc|    real_esrgan_int8_1b.bmodel       |      1.23       |      0.46       |      32.58      |     102.21      |
|   SE7-32    |real_esrgan_bmcv.soc|    real_esrgan_int8_4b.bmodel       |      1.11       |      0.43       |      31.70      |      98.82      |
|    SE9-8    |real_esrgan_opencv.py|    real_esrgan_fp32_1b.bmodel      |      35.20      |      23.67      |     3803.45     |      94.61      |
|    SE9-8    |real_esrgan_opencv.py|    real_esrgan_fp16_1b.bmodel      |      33.24      |      23.37      |     512.38      |      96.68      |
|    SE9-8    |real_esrgan_opencv.py|    real_esrgan_int8_1b.bmodel      |      23.94      |      23.70      |     536.92      |      96.74      |
|    SE9-8    |real_esrgan_opencv.py|    real_esrgan_int8_4b.bmodel      |      21.02      |      24.59      |     544.08      |      190.15     |
|    SE9-8    |real_esrgan_bmcv.py|    real_esrgan_fp32_1b.bmodel        |      22.99      |      3.82       |     3758.30     |     132.90      |
|    SE9-8    |real_esrgan_bmcv.py|    real_esrgan_fp16_1b.bmodel        |      20.30      |      3.79       |     467.83      |     135.42      |
|    SE9-8    |real_esrgan_bmcv.py|    real_esrgan_int8_1b.bmodel        |      3.46       |      3.47       |     125.32      |      76.28      |
|    SE9-8    |real_esrgan_bmcv.py|    real_esrgan_int8_4b.bmodel        |      8.95       |      3.01       |     121.99      |      99.81      |
|    SE9-8    |real_esrgan_bmcv.soc|    real_esrgan_fp32_1b.bmodel       |      15.14      |      1.60       |     3742.03     |     109.67      |
|    SE9-8    |real_esrgan_bmcv.soc|    real_esrgan_fp16_1b.bmodel       |      4.92       |      1.61       |     451.65      |     110.71      |
|    SE9-8    |real_esrgan_bmcv.soc|    real_esrgan_int8_1b.bmodel       |      4.99       |      1.59       |     120.86      |     123.16      |
|    SE9-8    |real_esrgan_bmcv.soc|    real_esrgan_int8_4b.bmodel       |      4.65       |      1.50       |     118.23      |     123.56      |
|   SE9-16    |real_esrgan_opencv.py|    real_esrgan_fp32_1b.bmodel      |      22.25      |      23.44      |     3791.06     |      87.81      |
|   SE9-16    |real_esrgan_opencv.py|    real_esrgan_fp16_1b.bmodel      |      14.34      |      23.80      |     500.06      |      87.83      |
|   SE9-16    |real_esrgan_opencv.py|    real_esrgan_int8_1b.bmodel      |      14.30      |      23.49      |     548.05      |      87.49      |
|   SE9-16    |real_esrgan_opencv.py|    real_esrgan_int8_4b.bmodel      |      18.25      |      23.48      |     545.85      |     137.49      |
|   SE9-16    |real_esrgan_bmcv.py|    real_esrgan_fp32_1b.bmodel        |      5.18       |      4.09       |     3746.08     |     131.82      |
|   SE9-16    |real_esrgan_bmcv.py|    real_esrgan_fp16_1b.bmodel        |      3.61       |      4.09       |     455.72      |     132.93      |
|   SE9-16    |real_esrgan_bmcv.py|    real_esrgan_int8_1b.bmodel        |      3.57       |      3.69       |     120.61      |      76.14      |
|   SE9-16    |real_esrgan_bmcv.py|    real_esrgan_int8_4b.bmodel        |      3.10       |      3.27       |     118.65      |      92.00      |
|   SE9-16    |real_esrgan_bmcv.soc|    real_esrgan_fp32_1b.bmodel       |      4.29       |      1.61       |     3729.59     |      68.16      |
|   SE9-16    |real_esrgan_bmcv.soc|    real_esrgan_fp16_1b.bmodel       |      3.78       |      1.62       |     439.28      |      65.37      |
|   SE9-16    |real_esrgan_bmcv.soc|    real_esrgan_int8_1b.bmodel       |      3.76       |      1.59       |     116.18      |      80.84      |
|   SE9-16    |real_esrgan_bmcv.soc|    real_esrgan_int8_4b.bmodel       |      3.67       |      1.49       |     114.54      |      78.07      |
|   SE9-16    |real_esrgan_opencv.py| real_esrgan_fp32_1b_2core.bmodel   |      14.25      |      23.74      |     1946.79     |      87.87      |
|   SE9-16    |real_esrgan_opencv.py| real_esrgan_fp16_1b_2core.bmodel   |      14.30      |      23.32      |     292.34      |      88.20      |
|   SE9-16    |real_esrgan_opencv.py| real_esrgan_int8_1b_2core.bmodel   |      14.28      |      23.46      |     519.40      |      87.98      |
|   SE9-16    |real_esrgan_opencv.py| real_esrgan_int8_4b_2core.bmodel   |      22.25      |      23.60      |     492.16      |     153.62      |
|   SE9-16    |real_esrgan_bmcv.py| real_esrgan_fp32_1b_2core.bmodel     |      17.72      |      3.99       |     1901.91     |     132.33      |
|   SE9-16    |real_esrgan_bmcv.py| real_esrgan_fp16_1b_2core.bmodel     |      3.53       |      4.03       |     247.17      |     132.53      |
|   SE9-16    |real_esrgan_bmcv.py| real_esrgan_int8_1b_2core.bmodel     |      3.55       |      3.86       |      92.06      |      75.69      |
|   SE9-16    |real_esrgan_bmcv.py| real_esrgan_int8_4b_2core.bmodel     |      3.07       |      3.23       |      64.27      |      85.18      |
|   SE9-16    |real_esrgan_bmcv.soc| real_esrgan_fp32_1b_2core.bmodel    |      4.68       |      1.61       |     1885.23     |      65.61      |
|   SE9-16    |real_esrgan_bmcv.soc| real_esrgan_fp16_1b_2core.bmodel    |      4.14       |      1.62       |     230.77      |      67.16      |
|   SE9-16    |real_esrgan_bmcv.soc| real_esrgan_int8_1b_2core.bmodel    |      4.12       |      1.59       |      87.69      |      81.85      |
|   SE9-16    |real_esrgan_bmcv.soc| real_esrgan_int8_4b_2core.bmodel    |      4.17       |      1.49       |      60.19      |      77.95      |
> **测试说明**：  
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. SE5-16/SE7-16的主控处理器均为8核 ARM A53 42320 DMIPS @2.3GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异，不同的阈值对后处理时间影响较大。 

## 8. FAQ
请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。