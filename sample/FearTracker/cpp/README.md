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
    * [3.3 输出视频](#33-输出视频)

cpp目录下提供C++例程以供参考使用，说明如下：
| 序号  | C++例程            | 说明                                        |
| ---- | ------------------ | ------------------------------------------- |
| 1    | fear_tracker_opencv | 使用OpenCV前处理、BMRT推理、OpenCV后处理     |


## 1. 环境准备
### 1.1 x86/arm/riscv PCIe平台
如果您在x86/arm/riscv平台安装了PCIe加速卡（如SC系列加速卡），可以直接使用它作为开发环境和运行环境。您需要安装libsophon、sophon-opencv和sophon-ffmpeg，具体步骤可参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)或[riscv-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#6-riscv-pcie平台的开发和运行环境搭建)。

### 1.2 SoC平台
如果您使用SoC平台（如SE、SM系列边缘设备），刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包，可直接使用它作为运行环境。通常还需要一台x86主机作为开发环境，用于交叉编译C++程序。

## 2. 程序编译
### 2.1 x86/arm/riscv PCIe平台
可以直接在PCIe平台上编译程序：

```bash
cd fear_tracker_opencv
mkdir build && cd build
cmake .. && make # 生成fear_tracker_opencv.pcie
cd ..
```
编译完成后，会在fear_tracker_opencv目录下生成fear_tracker_opencv.pcie。

### 2.2 SoC平台
通常在x86主机上交叉编译程序，您需要在x86主机上使用SOPHON SDK搭建交叉编译环境，将程序所依赖的头文件和库文件打包至soc-sdk目录中，具体请参考[交叉编译环境搭建](../../../docs/Environment_Install_Guide.md#41-交叉编译环境搭建)。本例程主要依赖libsophon、sophon-opencv和sophon-ffmpeg运行库包。

交叉编译环境搭建好后，使用交叉编译工具链编译生成可执行文件：

```bash
cd cpp/fear_tracker_opencv
mkdir build && cd build
#请根据实际情况修改-DSDK的路径，需使用绝对路径。
cmake -DTARGET_ARCH=soc -DSDK=/path_to_sdk/soc-sdk ..  
make
cd ..
```
编译完成后，会在fear_tracker_opencv目录下生成fear_tracker_opencv.soc。

SoC平台也可以直接在板子上原生编译：

```bash
cd cpp/fear_tracker_opencv
mkdir build && cd build
cmake -DTARGET_ARCH=soc -DSDK=/opt/sophon/libsophon-current .. && make
cd ..
```

## 3. 推理测试
对于PCIe平台，可以直接在PCIe平台上推理测试；对于SoC平台，需将交叉编译生成的可执行文件及所需的模型、测试数据拷贝到SoC平台中测试。测试的参数及运行方式是一致的，下面主要以SoC模式进行介绍。

### 3.1 参数说明
可执行程序默认有一套参数，请注意根据实际情况进行传参，具体参数说明如下：

```bash
usage:./fear_tracker_opencv.soc [params]

        --bmodel (value:../../models/BM1684X/feartracker_fp16_1b.bmodel)
                bmodel file path
        --dev_id (value:0)
                TPU device id
        --help (value:0)
                print help information.
        --input
                input video file path (required)
        --initial_bbox
                initial bbox: x,y,w,h (e.g. 163,53,45,174) (required)
        --output
                output video file path (optional)
```
**注意：** CPP传参与python不同，需要用等于号，例如`./fear_tracker_opencv.soc --bmodel=xxx`。

### 3.2 测试视频
视频测试实例如下，需指定输入视频文件和初始边界框：
```bash
./fear_tracker_opencv.soc --input=../../datasets/test.mp4 --initial_bbox=163,53,45,174 --bmodel=../../models/BM1684X/feartracker_fp16_1b.bmodel --dev_id=0
```
测试结束后，会在终端打印每帧跟踪到的边界框和平均推理时间。

### 3.3 输出视频
可选地，可以将跟踪结果保存为输出视频：
```bash
./fear_tracker_opencv.soc --input=../../datasets/test.mp4 --initial_bbox=123,45,67,89 --bmodel=../../models/BM1684X/feartracker_fp16_1b.bmodel --output=results/output.mp4 --dev_id=0
```
测试结束后，跟踪结果（绿色边界框）会保存在`--output`指定的视频文件中。
