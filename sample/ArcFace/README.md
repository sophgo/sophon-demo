# ArcFace

## 目录

- [1. 简介](#1-简介)
- [2. 特性](#2-特性)
  - [2.1 目录结构说明](#21-目录结构说明)
  - [2.2 SDK特性](#22-sdk特性)
- [3. 数据准备与模型编译](#3-数据准备与模型编译)
  - [3.1 数据准备](#31-数据准备)
  - [3.2 模型编译](#32-模型编译)
- [4. 例程测试](#4-例程测试)
- [5. 精度测试](#5-精度测试)
  - [5.1 测试方法](#51-测试方法)
  - [5.2 测试结果](#52-测试结果)
- [6. 性能测试](#6-性能测试)
  - [6.1 bmrt_test](#61-bmrt_test)
  - [6.2 程序运行性能](#62-程序运行性能)
- [7. FAQ](#7-faq)

## 1. 简介

ArcFace（IR-SE-ResNet50）人脸识别模型，基于[insightface](https://github.com/deepinsight/insightface)的buffalo_l预训练权重（w600k_r50），用于提取512维人脸特征嵌入向量。目前已适配BM1684X，支持在SOPHON BM1684X上进行推理测试。

**模型信息：**
- 输入：RGB图像，尺寸112x112，值域[0, 255]
- 预处理（嵌入BModel）：`(pixel - 127.5) × 0.0078125`
- 输出：512维浮点嵌入向量（经L2归一化）
- **网络结构**：IR-SE-ResNet50（BatchNorm → Gemm → L2 Norm）

## 2. 特性

### 2.1 目录结构说明
```bash
├── cpp                   # 存放C++例程及其README
|   ├──README.md
|   └──arcface_bmcv       # C++例程（原生bmrt API）
├── python                # 存放Python例程及其README
|   ├──README.md
|   └──arcface_bmcv.py    # Python例程（sophon.sail）
├── README.md             # 本例程的中文指南
├── scripts               # 存放模型编译、数据下载等shell脚本
|   ├── download.sh
|   ├── gen_fp32bmodel_mlir.sh
|   ├── gen_fp16bmodel_mlir.sh
|   └── gen_int8bmodel_mlir.sh
├── models                # 存放模型文件（通过download.sh下载）
└── datasets              # 存放测试/标定数据（通过download.sh下载）
```

### 2.2 SDK特性
- 支持BM1684X(x86 PCIe、SoC)
- 支持FP32、FP16(BM1684X)、INT8模型编译和推理
- 支持C++、Python推理
- 支持图片目录批量测试
- 输出512维L2归一化人脸嵌入向量

## 3. 数据准备与模型编译

### 3.1 数据准备

本例程在`scripts`目录下提供了相关模型和数据的下载脚本`download.sh`，**如果您希望自己准备模型和数据集，可以跳过本小节，参考[3.2 模型编译](#32-模型编译)进行模型转换。**

```bash
chmod -R +x scripts/
./scripts/download.sh --all
```

`download.sh`默认只下载`datasets`，`models`可以通过指定参数分平台下载，参数如下：
```bash
--all      # 下载所有模型
--BM1684X  # 下载BM1684X的bmodel
--onnx     # 下载onnx
```

下载的模型包括：
```bash
models/
├── BM1684X                             # 在BM1684X上运行的模型
│   ├── arcface_resnet50_fp32_1b.bmodel   # FP32 1batch (172MB)
│   ├── arcface_resnet50_fp16_1b.bmodel   # FP16 1batch (89MB)
│   ├── arcface_resnet50_int8_1b.bmodel   # INT8 1batch (48MB)
│   └── arcface_resnet50_int8_4b.bmodel   # INT8 4batch (48MB)
└── onnx
    └── w600k_r50.onnx                  # ONNX模型
```

下载的数据包括：
```bash
./datasets
├── test                            # 测试数据集（5张人脸图像）
└── cali                            # 量化标定数据集（100张人脸图像）
```

### 3.2 模型编译

**如果您不编译模型，只想直接使用下载的数据集和模型，可以跳过本小节。**

源模型需要编译成BModel才能在SOPHON TPU上运行，源模型在编译前要导出成onnx模型。建议使用TPU-MLIR编译BModel，模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录，并使用本例程提供的脚本将onnx模型编译为BModel。脚本中命令的详细说明可参考《TPU-MLIR开发手册》(请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的SDK中获取)。

- 生成FP32 BModel

本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X**），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684x
```

执行上述命令会在`models/BM1684X`文件夹下生成转换好的FP32 BModel。

- 生成FP16 BModel

本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X**），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x
```

执行上述命令会在`models/BM1684X/`文件夹下生成转换好的FP16 BModel。

- 生成INT8 BModel

本例程在`scripts`目录下提供了量化INT8 BModel的脚本，请注意修改`gen_int8bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台（**支持BM1684X**），如：

```shell
./scripts/gen_int8bmodel_mlir.sh bm1684x
```

上述脚本会在`models/BM1684X`文件夹下生成转换好的INT8 BModel。

## 4. 例程测试
- [C++例程](./cpp/README.md)
- [Python例程](./python/README.md)

## 5. 精度测试
### 5.1 测试方法

首先，参考[C++例程](cpp/README.md#32-测试图片)或[Python例程](python/README.md#22-测试图片)推理要测试的数据集。然后，使用`tools`目录下的`eval_accuracy.py`脚本，将预测结果与参考输出进行比对，验证模型推理精度。具体的测试命令如下：
```bash
# 请根据实际情况修改文件路径
python3 tools/eval_accuracy.py --input datasets/test --bmodel models/BM1684X/arcface_resnet50_fp32_1b.bmodel --dev_id 0
```

### 5.2 测试结果
|   测试平台    |      测试程序       |      测试模型          | 备注 |
| ------------ | ---------------- | ---------------------- | --- |
|   SE7-32    |  arcface_bmcv.py  | arcface_resnet50_fp32_1b.bmodel | C++与Python输出一致（误差<1e-5） |
|   SE7-32    |  arcface_bmcv.py  | arcface_resnet50_fp16_1b.bmodel | 与FP32余弦相似度>0.999 |
|   SE7-32    |  arcface_bmcv.py  | arcface_resnet50_int8_1b.bmodel | - |
|   SE7-32    |  arcface_bmcv.py  | arcface_resnet50_int8_4b.bmodel | - |
|   SE7-32    | arcface_bmcv.soc  | arcface_resnet50_fp32_1b.bmodel | C++与Python输出一致（误差<1e-5） |

> **测试说明**：
> 1. FP32模型C++与Python输出完全一致，FP16/INT8模型存在可接受的精度损失；
> 2. 在搭载了相同TPU和SOPHONSDK的PCIe或SoC平台上，相同程序的精度一致，SE7系列对应BM1684X；
> 3. BM1684X的FP16/INT8与FP32嵌入向量保持高度一致（余弦相似度>0.999）。

## 6. 性能测试
### 6.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684X/arcface_resnet50_fp32_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

|    测试平台  |              测试模型           | calculate time(ms) |
| ----------- | -------------------------------| ----------------- |
|   SE7-32    | BM1684X/arcface_resnet50_fp32_1b.bmodel     |           14.180  |
|   SE7-32    | BM1684X/arcface_resnet50_fp16_1b.bmodel     |           2.243   |
|   SE7-32    | BM1684X/arcface_resnet50_int8_1b.bmodel     |           1.189   |
|   SE7-32    | BM1684X/arcface_resnet50_int8_4b.bmodel     |           2.448 (4 images) |

> **测试说明**：
> 1. 性能测试结果具有一定的波动性；
> 2. INT8 4batch模型的`calculate time`为4张图像的总推理时间；
> 3. SoC和PCIe的测试结果基本一致。

### 6.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++和Python例程打印的时间已经折算为单张图片的处理时间。

在不同的测试平台上，使用不同的例程、模型测试，性能测试结果如下：
|    测试平台  |     测试程序      |        测试模型        |decode_time|preprocess_time|inference_time|postprocess_time|
| ----------- | ---------------- | ---------------------- | -------- | --------- | --------- | --------- |
|   SE7-32    | arcface_bmcv.py   | arcface_resnet50_fp32_1b.bmodel |     0.36       |     2.58       |     14.43      |     0.09       |
|   SE7-32    | arcface_bmcv.py   | arcface_resnet50_fp16_1b.bmodel |     0.36       |     2.56       |     2.46       |     0.09       |
|   SE7-32    | arcface_bmcv.py   | arcface_resnet50_int8_1b.bmodel |     0.36       |     2.58       |     1.42       |     0.09       |
|   SE7-32    | arcface_bmcv.py   | arcface_resnet50_int8_4b.bmodel |     0.36       |     2.51       |     1.20       |     0.09       |
|   SE7-32    |arcface_bmcv.soc   | arcface_resnet50_fp32_1b.bmodel |     5.77       |     0.19       |     14.04      |     0.02       |
|   SE7-32    |arcface_bmcv.soc   | arcface_resnet50_fp16_1b.bmodel |     0.68       |     0.19       |     2.12       |     0.02       |
|   SE7-32    |arcface_bmcv.soc   | arcface_resnet50_int8_1b.bmodel |     0.67       |     0.19       |     1.05       |     0.02       |
|   SE7-32    |arcface_bmcv.soc   | arcface_resnet50_int8_4b.bmodel |     0.74       |     0.22       |     1.44       |     0.01       |

> **测试说明**：
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. SE7-32的主控处理器为8核CA53@2.3GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 4. 图片分辨率对解码时间影响较大，不同的测试图片可能存在较大差异。

## 7. FAQ
请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。以下是ArcFace例程特定的FAQ：

**Q: C++ INT8 4b模型为何前4张图片输出相同嵌入向量？**

A: 这是一个已知的C++批处理实现问题，bm_image/tensor内存布局在INT8 4batch模式下存在对齐差异。Python sail接口的INT8 4b工作正常，batch内各图像输出独立且正确。建议生产环境优先使用Python接口进行多batch推理。

**Q: 预处理参数是否需要手动设置？**

A: 不需要。本例程将`(pixel - 127.5) * 0.0078125`归一化嵌入到bmodel编译阶段（TPU-MLIR的mean/scale参数），推理时C++和Python代码均设置为passthrough，由TPU硬件内部完成归一化。

**Q: 如何自定义人脸比对？**

A: 使用模型输出的512维归一化嵌入向量，通过余弦相似度计算距离：

```python
import numpy as np
similarity = np.dot(emb1, emb2)  # 归一化后等价于余弦相似度
threshold = 0.3  # 可根据实际场景调整
is_same = similarity > threshold
```
