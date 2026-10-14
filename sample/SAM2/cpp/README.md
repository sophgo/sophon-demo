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
    * [3.2 测试图片](#32-测试图片)

cpp目录下提供C++例程以供参考使用，说明如下：
| 序号  | C++例程          | 说明                                                       |
| ---- | ---------------- | ---------------------------------------------------------- |
| 1    | sam2_opencv      | 使用OpenCV前处理、BMRT推理、OpenCV后处理，支持图片和点/框输入 |

> 与Python例程类似，C++例程需要预先准备好encoder和decoder两个bmodel，运行时分别加载。模型准备方式见[准备模型与数据](../README.md#3-数据准备与模型编译)。


## 1. 环境准备
### 1.1 x86/arm/riscv PCIe平台
如果您在x86/arm/riscv平台安装了PCIe加速卡（如SC系列加速卡），可以直接使用它作为开发环境和运行环境。您需要安装libsophon、sophon-opencv和sophon-ffmpeg，具体步骤可参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)或[riscv-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#6-riscv-pcie平台的开发和运行环境搭建)。

### 1.2 SoC平台
如果您使用SoC平台（如SE、SM系列边缘设备），刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包，可直接使用它作为运行环境。通常还需要一台x86主机作为开发环境，用于交叉编译C++程序。

## 2. 程序编译
### 2.1 x86/arm/riscv PCIe平台
可以直接在PCIe平台上编译程序：

```bash
cd sam2_opencv
mkdir build && cd build
cmake .. && make # 生成sam2_opencv.pcie
cd ..
```
编译完成后，会在sam2_opencv目录下生成sam2_opencv.pcie。

### 2.2 SoC平台
通常在x86主机上交叉编译程序，您需要在x86主机上使用SOPHON SDK搭建交叉编译环境，将程序所依赖的头文件和库文件打包至soc-sdk目录中，具体请参考[交叉编译环境搭建](../../../docs/Environment_Install_Guide.md#41-交叉编译环境搭建)。本例程主要依赖libsophon、sophon-opencv运行库包。

交叉编译环境搭建好后，使用交叉编译工具链编译生成可执行文件：

```bash
cd cpp/sam2_opencv
mkdir build && cd build
#请根据实际情况修改-DSDK的路径，需使用绝对路径。
cmake -DTARGET_ARCH=soc -DSDK=/path_to_sdk/soc-sdk ..  
make
cd ..
```
编译完成后，会在sam2_opencv目录下生成sam2_opencv.soc。

SoC平台也可以直接在板子上原生编译：

```bash
cd cpp/sam2_opencv
mkdir build && cd build
cmake -DTARGET_ARCH=soc -DSDK=/opt/sophon/libsophon-current .. && make
cd ..
```

## 3. 推理测试
对于PCIe平台，可以直接在PCIe平台上推理测试；对于SoC平台，需将交叉编译生成的可执行文件及所需的模型、测试数据拷贝到SoC平台中测试。测试的参数及运行方式是一致的，下面主要以SoC模式进行介绍。

### 3.1 参数说明
可执行程序默认有一套参数，请注意根据实际情况进行传参，具体参数说明如下：

```bash
usage:./sam2_opencv.soc [params]

        --encoder_bmodel (value:../../models/BM1684X/image_encoder/sam2_encoder_f16_1b.bmodel)
                encoder bmodel file path
        --decoder_bmodel (value:../../models/BM1684X/image_decoder/sam2_decoder_f16_1b.bmodel)
                decoder bmodel file path
        --input_image (value:../../datasets/images/truck.jpg)
                input image path
        --input_point (value:500,375)
                point x,y or box x1,y1,x2,y2
        --output
                output image path (optional)
        --dev_id (value:0)
                TPU device id
        --help (value:0)
                print help information.
```
**注意：** CPP传参与python不同，需要用等于号，例如`./sam2_opencv.soc --input_point=500,375`。

### 3.2 测试图片
图片测试实例如下，通过`--input_point`指定单点提示（格式为`x,y`）或框提示（格式为`x1,y1,x2,y2`）：

```bash
./sam2_opencv.soc --encoder_bmodel=../../models/BM1684X/image_encoder/sam2_encoder_f16_1b.bmodel --decoder_bmodel=../../models/BM1684X/image_decoder/sam2_decoder_f16_1b.bmodel --input_image=../../datasets/images/truck.jpg --input_point=500,375 --output=results/cpp_output.jpg --dev_id=0
```

运行后会在终端打印各mask的IoU置信度和前景像素数，并将前三个mask分别叠加在原图上保存为输出图片。单点提示示例输出：

```
Got 3 masks
  Mask 0: score=0.0431213 pixels=633699
  Mask 1: score=0.9375 pixels=22175
  Mask 2: score=0.499268 pixels=47604
Saved: results/cpp_output_mask0.jpg
Saved: results/cpp_output_mask1.jpg
Saved: results/cpp_output_mask2.jpg
```

其中Mask 1置信度最高（score≈0.94），为最佳分割结果，与Python例程结果（score≈0.9429，pixels≈22279）基本一致。

框提示测试实例如下：

```bash
./sam2_opencv.soc --encoder_bmodel=../../models/BM1684X/image_encoder/sam2_encoder_f16_1b.bmodel --decoder_bmodel=../../models/BM1684X/image_decoder/sam2_decoder_f16_1b.bmodel --input_image=../../datasets/images/truck.jpg --input_point=425,280,745,595 --output=results/cpp_output_box.jpg --dev_id=0
```
