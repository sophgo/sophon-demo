<font size=5> C3D </font>
- [1. 简介](#1-简介)
- [2. 特性](#2-特性)
- [3. 准备模型与数据](#3-准备模型与数据)
- [4. 模型编译](#4-模型编译)
  - [4.1 TPU-NNTC编译BModel](#41-tpu-nntc编译bmodel)
  - [4.2 TPU-MLIR编译BModel](#42-tpu-mlir编译bmodel)
- [5. 例程测试](#5-例程测试)
- [6. 精度测试](#6-精度测试)
  - [6.1 测试方法](#61-测试方法)
  - [6.2 测试结果](#62-测试结果)
- [7. 性能测试](#7-性能测试)
  - [7.1 bmrt\_test](#71-bmrt_test)
  - [7.2 程序运行性能](#72-程序运行性能)
- [8. FAQ](#8-faq)



## 1. 简介
C3D是使用三维卷积进行视频动作识别的开荒者，论文链接：[Learning Spatiotemporal Features with 3D Convolutional Networks](https://arxiv.org/abs/1412.0767v4)。

本例程对[MMAction的C3D_UCF101模型](https://mmaction2.readthedocs.io/zh_CN/latest/recognition_models.html)进行了移植，在相同的预处理流程下可以做到精度对齐。
## 2. 特性
* 支持BM1684X(x86 PCIe、SoC)和BM1684(x86 PCIe、SoC、arm PCIe)
* 支持FP32、FP16(BM1684X)、INT8模型编译和推理
* 支持基于BMCV和OpenCV预处理的C++推理
* 支持基于OpenCV预处理的Python推理
* 支持单batch和多batch模型推理
* 支持视频文件夹测试
## 3. 准备模型与数据
如果您使用BM1684芯片，建议使用TPU-NNTC编译BModel，Pytorch模型在编译前要导出成torchscript模型或onnx模型；如果您使用BM1684X芯片，建议使用TPU-MLIR编译BModel，Pytorch模型在编译前要导出成onnx模型。

本例程在`scripts`目录下提供了**所有相关的模型和数据集**的下载脚本`download.sh`，您也可以自己准备模型和数据集，并参考[4. 模型转换](#4-模型转换)进行模型转换。

**如果您有自己训练的Pytorch C3D模型**，您可以参考tools/c3d_transform.py，**自行修改源模型路径和模型网络的层名，确保能够加载您的参数**，以成功转换torchscript和onnx模型。同时，您需要准备用于测试的数据集，如果量化模型，还要准备用于量化的数据集。

```bash
# 安装unzip，若已安装请跳过
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```
执行后，模型保存在`models`，数据集在`datasets`

下载的模型包括：
```
./models
├── BM1684
│   ├── c3d_fp32_1b.bmodel   # 使用TPU-NNTC编译，用于BM1684的FP32 BModel，batch_size=1
│   ├── c3d_fp32_4b.bmodel   # 使用TPU-NNTC编译，用于BM1684的FP32 BModel，batch_size=4
│   ├── c3d_int8_1b.bmodel   # 使用TPU-NNTC编译，用于BM1684的INT8 BModel，batch_size=1
│   └── c3d_int8_4b.bmodel   # 使用TPU-NNTC编译，用于BM1684的INT8 BModel，batch_size=4
├── BM1684X
│   ├── c3d_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=1
│   ├── c3d_fp32_4b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=4
│   ├── c3d_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=1
│   ├── c3d_fp16_4b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=4
│   ├── c3d_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=1
│   └── c3d_int8_4b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=4
│── torch
│   └── c3d_ucf101.pt        # trace后的torchscript模型
└── onnx
    └── c3d_ucf101.onnx      # 导出的onnx动态模型       
```
下载的数据包括：
```
./datasets/UCF_test_01       #UCF101的一个测试子集。
```



## 4. 模型编译
导出的模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。如果您使用BM1684芯片，建议使用TPU-NNTC编译BModel；如果您使用BM1684X芯片，建议使用TPU-MLIR编译BModel。

### 4.1 TPU-NNTC编译BModel
模型编译前需要安装TPU-NNTC，具体可参考[TPU-NNTC环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-nntc环境搭建)。安装好后需在TPU-NNTC环境中进入例程目录。

- 生成FP32 BModel

使用TPU-NNTC将trace后的torchscript模型编译为FP32 BModel，具体方法可参考《TPU-NNTC开发参考手册》的“BMNETP 使用”(请从[算能官网](https://developer.sophgo.com/site/index/material/28/all.html)相应版本的SDK中获取)。

​本例程在`scripts`目录下提供了TPU-NNTC编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_nntc.sh`中的torchscript模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（支持BM1684和BM1684X），如：

```bash
./scripts/gen_fp32bmodel_nntc.sh BM1684
```

​执行上述命令会在`models/BM1684/`下生成`c3d_fp32_1b.bmodel`等文件，即转换好的FP32 BModel。

- 生成INT8 BModel

使用TPU-NNTC量化torchscript模型的方法可参考《TPU-NNTC开发参考手册》的“模型量化”(请从[算能官网](https://developer.sophgo.com/site/index/material/28/all.html)相应版本的SDK中获取)，以及[模型量化注意事项](../../docs/Calibration_Guide.md#1-注意事项)。

​本例程在`scripts`目录下提供了TPU-NNTC量化INT8 BModel的脚本，请注意修改`gen_int8bmodel_nntc.sh`中的torchscript模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台，如：

```shell
./scripts/gen_int8bmodel_nntc.sh BM1684
```

​上述脚本会在`datasets/`下生成`cali_set_lmdb/`量化数据集，在`models/BM1684`下生成`c3d_int8_1b.bmodel`等文件，即转换好的INT8 BModel。

**如果您不使用本例程的数据集**，本例程在`tools`目录下提供了准备lmdb数据的python脚本，用户可以根据脚本自己生成lmdb量化数据集。
```bash
cd tools
python3 c3d_lmdb.py --input_path ../datasets/UCF_test_01 #for tpu-nntc, 需要在docker内进行。
```
执行后，会在datasets目录下产生`cali_set_lmdb`文件夹，可以作为量化模型使用的数据集。
### 4.2 TPU-MLIR编译BModel
模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#2-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index/material/31/all.html)相应版本的SDK中获取)。

- 生成FP32 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（支持BM1684X），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684x
```

​执行上述命令会在`models/BM1684X/`下生成`c3d_fp32_1b.bmodel`等文件，即转换好的FP32 BModel。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（支持BM1684X），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x
```

​执行上述命令会在`models/BM1684X/`下生成`c3d_fp16_1b.bmodel`等文件，即转换好的FP16 BModel。

- 生成INT8 BModel

​本例程在`scripts`目录下提供了量化INT8 BModel的脚本，请注意修改`gen_int8bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台（支持BM1684X），如：

```shell
./scripts/gen_int8bmodel_mlir.sh bm1684x
```

​上述脚本会在`datasets/`下生成`cali_set_npy/`量化数据集，在`models/BM1684X`下生成`c3d_int8_1b.bmodel`等文件，即转换好的INT8 BModel。

**如果您不使用本例程的数据集**，本例程在`tools`目录下提供了准备npy数据的python脚本，用户可以根据脚本自己准备npy格式量化数据集。
```bash
cd tools
python3 c3d_npy.py --input_path ../datasets/UCF_test_01 #for tpu-mlir
```
执行后，会在datasets目录下产生`cali_set_npy`文件夹，可以作为量化模型使用的数据集。

## 5. 例程测试
- [C++例程](./cpp/README.md)
- [Python例程](./python/README.md)

## 6. 精度测试
### 6.1 测试方法

首先，参考[C++例程](cpp/README.md#32-测试视频理解数据集)或[Python例程](python/README.md#22-测试视频理解数据集)推理要测试的数据集，生成预测的json文件。
然后，使用`tools`目录下的`eval_ucf.py`脚本，将测试生成的json文件与测试集标签json文件进行对比，计算出准确率信息，命令如下：
```bash
# 请根据实际情况修改程序路径和json文件路径
python3 tools/eval_ucf.py --gt_path datasets/ground_truth.json --result_json cpp/c3d_bmcv/results/c3d_fp32_1b.bmodel_bmcv_cpp.json
```
### 6.2 测试结果
根据本例程提供的数据集，测试结果如下：
|   测试平台    |      测试程序     |    测试模型        | ACC |
| ------------ | ---------------- | ------------------ | --- |
| BM1684 PCIe  | c3d_opencv.py    | c3d_fp32_1b.bmodel |0.715|
| BM1684 PCIe  | c3d_opencv.py    | c3d_int8_1b.bmodel |0.691|
| BM1684 PCIe  | c3d_opencv.pcie  | c3d_fp32_1b.bmodel |0.715|
| BM1684 PCIe  | c3d_opencv.pcie  | c3d_int8_1b.bmodel |0.691|
| BM1684 PCIe  | c3d_bmcv.pcie    | c3d_fp32_1b.bmodel |0.715|
| BM1684 PCIe  | c3d_bmcv.pcie    | c3d_int8_1b.bmodel |0.693|
| BM1684X PCIe | c3d_opencv.py    | c3d_fp32_1b.bmodel |0.715|
| BM1684X PCIe | c3d_opencv.py    | c3d_fp16_1b.bmodel |0.715|
| BM1684X PCIe | c3d_opencv.py    | c3d_int8_1b.bmodel |0.713|
| BM1684X PCIe | c3d_opencv.pcie  | c3d_fp32_1b.bmodel |0.715|
| BM1684X PCIe | c3d_opencv.pcie  | c3d_fp16_1b.bmodel |0.715|
| BM1684X PCIe | c3d_opencv.pcie  | c3d_int8_1b.bmodel |0.713|
| BM1684X PCIe | c3d_bmcv.pcie    | c3d_fp32_1b.bmodel |0.715|
| BM1684X PCIe | c3d_bmcv.pcie    | c3d_fp16_1b.bmodel |0.715|
| BM1684X PCIe | c3d_bmcv.pcie    | c3d_int8_1b.bmodel |0.713|

> **测试说明**：  
1. batch_size=4和batch_size=1的模型精度一致；
2. 由于不同平台的系统存在解码差异，SoC和PCIe模式下相同模型的准确率也会存在细微差异；

## 7. 性能测试
### 7.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径
bmrt_test --bmodel models/BM1684/c3d_fp32_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是理论推理时间。
测试各个模型的理论推理时间，结果如下：

|          测试模型           | calculate time(ms) |
| -------------------------- | ----------------- |
| BM1684/c3d_fp32_1b.bmodel  | 55.5           |
| BM1684/c3d_fp32_4b.bmodel  | 46.7           |
| BM1684/c3d_int8_1b.bmodel  | 43.0            |
| BM1684/c3d_int8_4b.bmodel  | 17.3             |
| BM1684X/c3d_fp32_1b.bmodel | 80.8              |
| BM1684X/c3d_fp32_4b.bmodel | 75.3              |
| BM1684X/c3d_fp16_1b.bmodel | 11.7               |
| BM1684X/c3d_fp16_4b.bmodel | 9.0               |
| BM1684X/c3d_int8_1b.bmodel | 10.1               |
| BM1684X/c3d_int8_4b.bmodel | 8.7               |

> **测试说明**：  
1. 性能测试结果具有一定的波动性；
2. `calculate time`已折算为每个视频平均推理时间。

### 7.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的视频解码时间、预处理时间、推理时间、后处理时间。C++例程打印的预处理时间、推理时间、后处理时间为整个batch处理的时间，需除以相应的batch size才是每个视频平均处理时间。

在不同的测试平台上，使用不同的例程、模型测试`datasets/test`，性能测试结果如下：
|    测试平台  |     测试程序      |      测试模型    |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ---------------- | ---------------- | -------- | --------- | --------- | --------- |
| BM1684 SoC  | c3d_opencv.py | c3d_fp32_1b.bmodel | 75.50    | 31.57  |   62.01 |  0.09   |
| BM1684 SoC  | c3d_opencv.py | c3d_fp32_4b.bmodel | 51.94    | 28.81  | 53.65   |  0.03   |
| BM1684 SoC  | c3d_opencv.py | c3d_int8_1b.bmodel | 71.45    | 30.89  | 49.61  |  0.10  |
| BM1684 SoC  | c3d_opencv.py | c3d_int8_4b.bmodel | 53.06    | 28.57   | 24.26  |  0.03   |
| BM1684 SoC  | c3d_bmcv.soc  | c3d_fp32_1b.bmodel | 79.53   |  3.59   |  55.30   |  0.02   |
| BM1684 SoC  | c3d_bmcv.soc  | c3d_fp32_4b.bmodel | 80.08   |  3.48   |  46.63  |   0.02      |
| BM1684 SoC  | c3d_bmcv.soc  | c3d_int8_1b.bmodel | 79.15   |  3.62   |  42.70  |  0.02   |
| BM1684 SoC  | c3d_bmcv.soc  | c3d_int8_4b.bmodel | 79.99   |  3.45   |  17.25   |  0.02   |
| BM1684 SoC  | c3d_opencv.soc  | c3d_fp32_1b.bmodel | 77.03   | 26.20   |  55.33   |  0.01   |
| BM1684 SoC  | c3d_opencv.soc  | c3d_fp32_4b.bmodel | 76.81   | 26.12    | 46.64   |  0.02    |
| BM1684 SoC  | c3d_opencv.soc  | c3d_int8_1b.bmodel | 75.70   | 26.87  |   42.74   |  0.01  |
| BM1684 SoC  | c3d_opencv.soc  | c3d_int8_4b.bmodel | 75.11   | 26.93   |  17.28  |   0.02   |
| BM1684X SoC | c3d_opencv.py | c3d_fp32_1b.bmodel |  67.19    | 31.01    |  88.76  |  0.10   |
| BM1684X SoC | c3d_opencv.py | c3d_fp32_4b.bmodel |  49.85      |  29.09  |  84.06  |   0.03  |
| BM1684X SoC | c3d_opencv.py | c3d_fp16_1b.bmodel |  65.62     |  31.07 |  19.59  |   0.10   |
| BM1684X SoC | c3d_opencv.py | c3d_fp16_4b.bmodel |  48.72      | 29.12  |  17.87  | 0.03   |
| BM1684X SoC | c3d_opencv.py | c3d_int8_1b.bmodel |  65.41      |  31.05  | 17.98   | 0.10   |
| BM1684X SoC | c3d_opencv.py | c3d_int8_4b.bmodel |  49.36     |  29.21  | 17.53   |  0.03    |
| BM1684X SoC | c3d_bmcv.soc  | c3d_fp32_1b.bmodel |  77.05     |   2.30   |  80.76  | 0.02    |
| BM1684X SoC | c3d_bmcv.soc  | c3d_fp32_4b.bmodel |   76.47      |  2.26   |  75.30  |  0.02   |
| BM1684X SoC | c3d_bmcv.soc  | c3d_fp16_1b.bmodel |   76.55    |  2.30  |  11.63   |   0.02     |
| BM1684X SoC | c3d_bmcv.soc  | c3d_fp16_4b.bmodel |   76.62    |  2.25  |  9.08    |   0.02    |
| BM1684X SoC | c3d_bmcv.soc  | c3d_int8_1b.bmodel |   77.35     |  2.31   | 10.03    |  0.02     |
| BM1684X SoC | c3d_bmcv.soc  | c3d_int8_4b.bmodel |   76.69    |  2.23  | 8.72  |   0.02   |
| BM1684X SoC | c3d_opencv.soc  | c3d_fp32_1b.bmodel |  72.81     |   26.76   |  80.76  | 0.02    |
| BM1684X SoC | c3d_opencv.soc  | c3d_fp32_4b.bmodel |   73.38      |  26.42  |  75.30  |  0.02   |
| BM1684X SoC | c3d_opencv.soc  | c3d_fp16_1b.bmodel |   73.70    |  26.86 |  11.63   |   0.02     |
| BM1684X SoC | c3d_opencv.soc  | c3d_fp16_4b.bmodel |   73.85    |  26.52  |  9.08    |   0.02    |
| BM1684X SoC | c3d_opencv.soc  | c3d_int8_1b.bmodel |   73.34     |  26.72   | 10.03    |  0.02     |
| BM1684X SoC | c3d_opencv.soc  | c3d_int8_4b.bmodel |   73.87    |  26.36  | 8.72  |   0.02   |


> **测试说明**：  
1. 时间单位均为毫秒(ms)，统计的时间均为每个视频平均处理时间；
2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
3. BM1684/1684X SoC的主控CPU均为8核 ARM A53 42320 DMIPS @2.3GHz，PCIe上的性能由于CPU的不同可能存在较大差异；
4. 帧分辨率对解码时间影响较大，不同的测试视频可能存在较大差异。 

## 8. FAQ
请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。