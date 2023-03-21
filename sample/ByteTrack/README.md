简体中文 ｜ [English](./docs/README_EN.md)
# ByteTrack

使用OpenCV和BMCV部署YOLOX+ByteTrack目标跟踪，包括程序的C++和Python版本。

- [ByteTrack](#bytetrack)
  - [1. 介绍](#1-介绍)
  - [2. 特性](#2-特性)
  - [3. 准备模型与数据](#3-准备模型与数据)
  - [4. 模型编译](#4-模型编译)
    - [4.1 TPU-NNTC编译BModel](#41-tpu-nntc编译bmodel)
    - [4.2 TPU-MLIR编译BModel](#42-tpu-mlir编译bmodel)
  - [5. 例程测试](#5-例程测试)
  - [6. 精度测试](#6-精度测试)
    - [6.1 测试方法](#61-测试方法)
    - [6.2 自动测试](#62-自动测试)
    - [6.3 测试结果](#63-测试结果)
  - [7. 性能测试](#7-性能测试)
    - [7.1 bmrt\_test](#71-bmrt_test)
    - [7.2 程序运行性能](#72-程序运行性能)
  - [8. FAQ](#8-faq)

## 1. 介绍
ByteTrack是一个简单、快速、强大的多目标跟踪器。
多目标跟踪(MOT)旨在估计视频中物体的边界框和标识。大多数方法通过关联检测分数高于阈值的检测框来获取标识。低检测分数的物体（如被遮挡的物体）会被丢弃，这会导致丢失真实物体和碎片化轨迹。为解决这个问题，我们提出了一种简单、有效和通用的关联方法，即通过关联每个检测框来跟踪，而不仅仅是高分数的那些。对于低分检测框，我们利用它们与轨迹片段的相似性来恢复真实物体，并过滤掉背景检测。

**论文** (https://arxiv.org/abs/2110.06864)

**源代码** (https://github.com/ifzhang/ByteTrack)

## 2. 特性
* 支持BM1684X(x86 PCIe、SoC)和BM1684(x86 PCIe、SoC、arm PCIe)
* 支持FP32模型编译和推理
* 支持基于BMCV预处理的C++推理
* 支持基于OpenCV和BMCV预处理的Python推理
* 支持图片和视频测试

## 3. 准备模型与数据
Pytorch的模型需要经过“torch.jit.trace”才能编译，trace的模型可以用于编译BModel。可以在[torch.jit.trace Guide](../../docs/torch.jit.trace_Guide.md)中找到trace的方法和原理。

同时，您需要准备一个用于测试的数据集，如果量化模型，还需要一个用于量化的数据集。

本例提供了相关模型和数据集的下载脚本“download.sh”在“scripts”目录下，运行后自动下载pt模型、数据集和BModel，即可以跳过第4章模型编译。您也可以使用下载的pt模型和量化数据集，或自己准备模型和数据集，并参考[4. 模型编译](#4-模型编译)进行模型转换以生成BModel。

```bash
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install unzip
cd ./scripts
chmod +x download.sh
./download.sh
```

执行后，模型将保存到“models/”目录下，测试视频和数据将保存到“datasets/”目录下。

下载的模型包括：
```
./models
├── BM1684
│   ├── bytetrack_s_fp32_1b.bmodel   # 使用TPU-NNTC编译，用于BM1684的FP32 BModel，batch_size=1
├── BM1684X
│   ├── bytetrack_s_st_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=1
└── onnx
    └── bytetrack_s.onnx             # 导出的onnx动态模型
```

下载的数据包括：
```
./datasets
├── sample.mp4                                # 测试视频
└── MOT15                                     # MOT15数据集
    └──  ADL-Rundle-6                         # 抽取MOT15中train目录下的ADL-Rundle-6
          ├── det                             # 检测对比数据
          ├── gt                              # 真实结果
          └── img1                            # 检测图片
```

## 4. 模型编译

导出的模型需要编译成BModel才能在sophon TPU上运行，如果使用下载好的BModel可跳过本节。如果您使用BM1684芯片，建议使用TPU-NNTC编译BModel；如果您使用BM1684X芯片，建议使用TPU-MLIR编译BModel。

### 4.1 TPU-NNTC编译BModel

模型编译前需要安装TPU-NNTC，具体可参考[TPU-NNTC环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-nntc环境搭建)。安装好后需在TPU-NNTC环境中进入例程目录。

- 生成FP32 BModel

使用TPU-NNTC将trace后的torchscript模型编译为FP32 BModel，也可以直接编译onnx模型，具体方法可参考《TPU-NNTC开发参考手册》”(请从[算能官网](https://developer.sophgo.com/site/index/material/28/all.html)相应版本的SDK中获取)。

​本例程在`scripts`目录下提供了TPU-NNTC编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_nntc.sh`中的模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（支持BM1684和BM1684X），如：

```bash
cd scripts/
chmod +x gen_fp32bmodel_nntc.sh
./gen_fp32bmodel_nntc.sh BM1684
```

​执行上述命令会在`models/BM1684/`下生成`bytetrack_s_fp32_1b.bmodel`文件，即转换好的FP32 BModel。

### 4.2 TPU-MLIR编译BModel
模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#2-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index/material/31/all.html)相应版本的SDK中获取)。

- 生成FP32 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（支持BM1684X），如：

```bash
cd scripts/
chmod +x gen_fp32bmodel_mlir.sh
./gen_fp32bmodel_mlir.sh bm1684x
```

​执行上述命令会在`models/BM1684X/`下生成`bytetrack_s_fp32_1b.bmodel`文件，即转换好的FP32 BModel。

## 5. 例程测试

- [C++例程](./cpp/README.md)
- [Python例程](./python/README.md)

## 6. 精度测试

### 6.1 测试方法

首先，参考[C++例程](cpp/README.md#41-测试MOT数据集)或[Python例程](python/README.md#31-测试MOT数据集)推理要测试的数据集，生成包含目标追踪结果的txt文件，注意修改数据集(datasets/MOT15/ADL-Rundle-6/img1)。
然后，使用`tools`目录下的`eval_mot.py`脚本，将测试生成的txt文件与测试集标签txt文件进行对比，计算出目标追踪的一系列评价指标，命令如下：
```bash
# 安装motmetrics，若已安装请跳过
pip3 install motmetrics
# 请根据实际情况修改程序路径和txt文件路径
    python3 ../tools/eval_mot.py \
      --ground_truths=../datasets/MOT15/ADL-Rundle-6/gt/gt.txt \
      --detections=../cpp/bytetrack_bmcv/results/img1_bytetrack_s_fp32_1b_cpp.txt
```

运行结果：
```bash
MOTA = -0.4791375524056698
     num_frames      IDF1       IDP       IDR      Rcll     Prcn    GT  MT  PT  ML    FP    FN  IDsw   FM      MOTA      MOTP
acc         525  0.049857  0.058351  0.043522  0.159114  0.21333  5009   0   7  17  2939  4212   258  562 -0.479138  0.342534
```

### 6.2 自动测试

此自动化测试脚本需要在带有PCIe加速卡的x86主机或Sophon SoC设备上执行。

依赖于python包'motmetrics'
```bash
# 安装motmetrics，若已安装请跳过
pip3 install motmetrics
```

准备好测试数据的BModel之后：

```bash
cd scripts
chmod +x ./auto_test.sh
./auto_test.sh ${platform} ${target} ${tpu_id} ${sail_dir}
```

其中，'platform'指平台（x86或soc），'target'是芯片型号（BM1684或BM1684X），'tpu_id'指TPU的ID（使用BM-SMI查看），'sail_dir'是SAIL的安装路径。如果最终输出为'Failed:'，则执行失败，否则表示成功。

例如，

```bash
./auto_test.sh soc BM1684 0 /opt/sophon/sophon-sail
```

在x86上，`auto_test.sh包括在cpp文件夹中编译和操作C++程序以及在Python文件夹中运行所有Python程序，以及操作MOT指标脚本。

在soc上，`auto_test.sh`包括在cpp文件夹中操作C++程序以及在Python文件夹中运行所有Python程序，以及操作MOT指标脚本。

要在x86上执行此脚本，请参考[x86-pcie平台的开发和运行环境搭建](../../docs/Environment_Install_Guide.md#2-x86-pcie平台的开发和运行环境搭建)，然后运行此脚本，其中${sail_dir}构建上述环境的sophon-sail安装路径，通常为/opt/sophon/sophon-sail。

要在SoC上执行此脚本，首先需要交叉编译ARM程序，请参考[交叉编译环境搭建](../../docs/Environment_Install_Guide.md#31-交叉编译环境搭建)，然后将生成的可执行文件移动到cpp文件夹中。之后，设置环境变量。

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/sophon/sophon-sail/lib
```

再次运行此脚本，其中${sail_dir}是为上述环境创建的build_soc/sophon-sail文件夹。

### 6.3 测试结果
这里使用目标检测模型bytetrack_s_fp32_1b.bmodel，使用数据集ADL-Rundle-6，记录MOTA作为精度指标，精度测试结果如下：

|   测试平台   |        测试程序       |           测试模型          | MOTA |
| ------------|   ----------------  | -------------------------- | ---- |
| BM1684 SoC  | bytetrack_bmcv.soc  | bytetrack_s_fp32_1b.bmodel | 47.9 |
| BM1684 SoC  | bytetrack_opencv.py | bytetrack_s_fp32_1b.bmodel | 44.1 |
| BM1684 SoC  | bytetrack_bmcv.py   | bytetrack_s_fp32_1b.bmodel | 37.1 |


## 7. 性能测试
### 7.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径
bmrt_test --bmodel models/BM1684/bytetrack_s_fp32_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间。
测试各个模型的理论推理时间，结果如下：

|           测试模型                  |  calculate time(ms) |
| -----------------------------      |  -----------------  |
| BM1684/bytetrack_s_fp32_1b.bmodel  |      40.50         |


> **测试说明**：
1. 性能测试结果具有一定的波动性；
2. `calculate time`已折算为平均每张图片的推理时间。

### 7.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的预处理时间、推理时间、后处理时间。

在不同的测试平台上，使用不同的例程、模型测试`datasets/MOT5/ADL-Rundle-6/img1`，性能测试结果如下：
|    测试平台  |     测试程序          |         测试模型            |preprocess_time|inference_time|postprocess_time|track_time| overall_time|
| ----------- | ----------------    |  ------------------------- | ------------- | ------------- | ------------ |  --------- | ---------- |
| BM1684 soc  | bytetrack_opencv.py | bytetrack_s_fp32_1b.bmodel |     214.70    |     54.59     |     7.87     |    10.26   |   359.31   |
| BM1684 soc  | bytetrack_bmcv.py   | bytetrack_s_fp32_1b.bmodel |     30.19     |     41.50     |     7.66     |     9.39   |    99.91   |
| BM1684 soc  | bytetrack_bmcv.soc  | bytetrack_s_fp32_1b.bmodel |     10.84     |     40.56     |     0.19     |     0.78   |    52.37   |



> **测试说明**：
1. 时间单位均为毫秒(ms)，preprocess_time、inference_time，postprocess_time是YOLOX探测器的处理时间，track_time是bytetrack算法更新tracker的时间，overall_time是处理一帧图像的时间；
2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
3. BM1684/1684X SoC的主控CPU均为8核 ARM A53 42320 DMIPS @2.3GHz，PCIe上的性能由于CPU的不同可能存在较大差异；
4. 图片分辨率对解码时间影响较大；

## 8. FAQ
请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。