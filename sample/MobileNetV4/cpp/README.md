# C++例程

## 目录

* [1. 环境准备](#1-环境准备)
    * [1.1 x86/arm/riscv PCIe平台](#11-x86armriscv-pcie平台)
    * [1.2 SoC平台](#12-soc平台)
* [2. 编译程序](#2-编译程序)
    * [2.1 x86/arm/riscv PCIe平台](#21-x86armriscv-pcie平台)
    * [2.2 SoC平台](#22-soc平台)
* [3. 推理测试](#3-推理测试)
    * [3.1 参数说明](#31-参数说明)
    * [3.2 测试图片](#32-测试图片)

cpp目录下提供了一系列C++例程，参考[YOLOv8_plus_cls](../../YOLOv8_plus_cls/cpp/README.md)，具体情况如下：

| 序号   | C++例程              | 说明                                |
| ---- | -------------------- | -----------------------------------  |
| 1    | mobilenetv4_bmcv     | 使用OpenCV解码、BMCV前处理、BMRT推理   |

## 1. 环境准备
### 1.1 x86/arm/riscv PCIe平台

如果您在x86/arm/riscv平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon和sophon-opencv，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)或[riscv-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#6-riscv-pcie平台的开发和运行环境搭建)。

### 1.2 SoC平台

如果您使用SoC平台（如SE、SM系列边缘设备），并使用它测试本例程，刷机后在`/opt/sophon/`下已经预装了相应的libsophon和sophon-opencv运行库包。

## 2. 编译程序
### 2.1 x86/arm/riscv PCIe平台
可以直接在PCIe平台上编译程序：

```bash
cd cpp/mobilenetv4_bmcv
mkdir build && cd build
cmake .. && make
cd ..
# 生成 mobilenetv4_bmcv.pcie
```

### 2.2 SoC平台
通常在x86主机上交叉编译程序，您需要在x86主机上使用SOPHON SDK搭建交叉编译环境，将程序所依赖的头文件和库文件打包至soc-sdk目录中，具体请参考[交叉编译环境搭建](../../../docs/Environment_Install_Guide.md#41-交叉编译环境搭建)。本例程主要依赖libsophon和sophon-opencv运行库包。

交叉编译环境搭建好后，使用交叉编译工具链编译生成可执行文件：

```bash
cd cpp/mobilenetv4_bmcv
mkdir build && cd build
cmake .. -DTARGET_ARCH=soc -DSDK=/path_to_sdk/soc-sdk && make
# 生成 mobilenetv4_bmcv.soc
```

## 3. 推理测试
### 3.1 参数说明
可执行程序默认有一套参数，请注意根据实际情况进行传参，`mobilenetv4_bmcv.pcie`参数说明如下：

```bash
Usage:
  --input: 测试图片路径，可输入整个图片文件夹的路径；
  --bmodel: 用于推理的bmodel路径，默认使用stage 0的网络进行推理；
  --dev_id: 用于推理的tpu设备id；
  --help: 输出帮助信息。
```

### 3.2 测试图片
图片测试实例如下，支持对整个图片文件夹进行测试：
```bash
./mobilenetv4_bmcv.pcie --input=../../datasets/imagenet_val_1k/img --bmodel=../../models/BM1684X/mobilenetv4_conv_medium_fp32_1b.bmodel --dev_id=0
```
测试结束后，会将预测结果保存在`results/mobilenetv4_conv_medium_fp32_1b.bmodel_img_bmcv_cpp_result.json`下，同时会打印预测结果、推理时间等信息。
