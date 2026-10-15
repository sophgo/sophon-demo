# C++例程

## 目录

* [1. 环境准备](#1-环境准备)
    * [1.1 x86/arm/riscv PCIe平台](#11-x86armriscv-pcie平台)
    * [1.2 SoC平台](#12-soc平台)
* [2. 程序编译](#2-程序编译)
    * [2.1 x86/arm/riscv PCIe平台](#21-x86armriscv-pcie平台)
    * [2.2 SoC平台](#22-soc平台)
* [3. 推理测试](#3-推理测试)
    * [3.1 参数说明](#31-参数说明)
    * [3.2 测试视频](#32-测试视频)
    * [3.3 测试图片](#33-测试图片)
    * [3.4 文本嵌入预计算](#34-文本嵌入预计算)

cpp目录下提供C++例程以供参考使用，说明如下：
| 序号  | C++例程            | 说明                                                       |
| ---- | ------------------ | ---------------------------------------------------------- |
| 1    | yoloworld_opencv   | 使用OpenCV前处理、BMRT推理、OpenCV后处理，支持图片和视频    |

> 与Python例程类似，C++例程需要预先计算好文本嵌入 `txt_feats`（`[1,80,512]` float32），运行时通过 `--txt_feats` 参数加载。文本嵌入的预计算方式见 [3.4 文本嵌入预计算](#34-文本嵌入预计算)。


## 1. 环境准备
### 1.1 x86/arm/riscv PCIe平台
如果您在x86/arm/riscv平台安装了PCIe加速卡（如SC系列加速卡），可以直接使用它作为开发环境和运行环境。您需要安装libsophon、sophon-opencv和sophon-ffmpeg，具体步骤可参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)或[riscv-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#6-riscv-pcie平台的开发和运行环境搭建)。

### 1.2 SoC平台
如果您使用SoC平台（如SE、SM系列边缘设备），刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包，可直接使用它作为运行环境。通常还需要一台x86主机作为开发环境，用于交叉编译C++程序。

## 2. 程序编译
### 2.1 x86/arm/riscv PCIe平台
可以直接在PCIe平台上编译程序：

```bash
cd yoloworld_opencv
mkdir build && cd build
cmake .. && make # 生成yoloworld_opencv.pcie
cd ..
```
编译完成后，会在yoloworld_opencv目录下生成yoloworld_opencv.pcie。

### 2.2 SoC平台
通常在x86主机上交叉编译程序，您需要在x86主机上使用SOPHON SDK搭建交叉编译环境，将程序所依赖的头文件和库文件打包至soc-sdk目录中，具体请参考[交叉编译环境搭建](../../../docs/Environment_Install_Guide.md#41-交叉编译环境搭建)。本例程主要依赖libsophon、sophon-opencv和sophon-ffmpeg运行库包。

交叉编译环境搭建好后，使用交叉编译工具链编译生成可执行文件：

```bash
cd cpp/yoloworld_opencv
mkdir build && cd build
#请根据实际情况修改-DSDK的路径，需使用绝对路径。
cmake -DTARGET_ARCH=soc -DSDK=/path_to_sdk/soc-sdk ..  
make
cd ..
```
编译完成后，会在yoloworld_opencv目录下生成yoloworld_opencv.soc。

SoC平台也可以直接在板子上原生编译：

```bash
cd cpp/yoloworld_opencv
mkdir build && cd build
cmake -DTARGET_ARCH=soc -DSDK=/opt/sophon/libsophon-current .. && make
cd ..
```

## 3. 推理测试
对于PCIe平台，可以直接在PCIe平台上推理测试；对于SoC平台，需将交叉编译生成的可执行文件及所需的模型、测试数据拷贝到SoC平台中测试。测试的参数及运行方式是一致的，下面主要以SoC模式进行介绍。

### 3.1 参数说明
可执行程序默认有一套参数，请注意根据实际情况进行传参，具体参数说明如下：

```bash
usage:./yoloworld_opencv.soc [params]

        --bmodel (value:../../models/BM1684X/yoloworld_fp16_1b.bmodel)
                bmodel file path
        --txt_feats (value:../../models/BM1684X/coco80_txt_feats.npy)
                precomputed txt_feats .npy [1,N,512]
        --class_names_file (value:../../datasets/coco.names)
                class names file (one name per line)
        --input
                input: image directory or video file (required)
        --output
                output video/image path (optional)
        --conf_thresh (value:0.25)
                confidence threshold
        --nms_thresh (value:0.7)
                nms threshold
        --dev_id (value:0)
                TPU device id
        --help (value:0)
                print help information.
```
**注意：** CPP传参与python不同，需要用等于号，例如`./yoloworld_opencv.soc --bmodel=xxx`。

### 3.2 测试视频
视频测试实例如下，需指定输入视频文件和预计算的txt_feats：
```bash
./yoloworld_opencv.soc --input=../../datasets/test_car_person_1080P.mp4 --bmodel=../../models/BM1684X/yoloworld_fp16_1b.bmodel --txt_feats=../../models/BM1684X/coco80_txt_feats.npy --class_names_file=../../datasets/coco.names --dev_id=0
```
测试结束后，会在终端打印每帧检测到的目标数量和平均推理时间。可选地，加上`--output=results/output.mp4`可将检测结果保存为输出视频。

### 3.3 测试图片
图片目录测试实例如下：
```bash
./yoloworld_opencv.soc --input=../../datasets/test --bmodel=../../models/BM1684X/yoloworld_fp16_1b.bmodel --txt_feats=../../models/BM1684X/coco80_txt_feats.npy --class_names_file=../../datasets/coco.names --output=results/images --dev_id=0
```

### 3.4 文本嵌入预计算
C++例程不内置CLIP文本编码和BPE分词，而是在启动时一次性加载预计算的文本嵌入。预计算方式（在Python环境下运行一次即可，`txt_feats`需与`class_names_file`中的类别顺序一致）：

```bash
cd python
# 使用tools目录下的小脚本，运行一次CLIP文本编码并导出 coco80_txt_feats.npy
python3 ../tools/export_txt_feats.py --bmodel=../models/BM1684X/yoloworld_fp16_1b.bmodel --clip_bmodel=../models/BM1684X/clip_text_vitb32_bm1684x_f16_1b.bmodel --class_names_file=../datasets/coco.names --output=../models/BM1684X/coco80_txt_feats.npy
```

导出的 `coco80_txt_feats.npy` 为 `[1, 80, 512]` float32（长度不足`num_classes`时自动补零），可在C++例程中通过 `--txt_feats` 参数复用，每帧推理时无需重复运行CLIP文本编码。
