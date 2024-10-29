# CAM++ C++例程

## 目录
- [CAM++ C++例程](#CAM++-C++例程)
  - [目录](#目录)
  - [1. 环境准备](#1-环境准备)
    - [1.1 x86/arm PCIe平台](#11-x86arm-pcie平台)
    - [1.2 SoC平台](#12-soc平台)
  - [2. 编译程序](#2-编译程序)
  - [3. 例程测试](#3-例程测试)
    - [3.1 参数说明](#31-参数说明)
    - [3.2 使用方式](#32-使用方式)
  - [4. kaldi编译](#3-kaldi编译)

## 1. 环境准备
### 1.1 x86/arm PCIe平台
如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。

- 请确保您的驱动及libsophon版本满足本例程的要求，具体请参考[简介](../README.md#1-简介)

### 1.2 SoC平台

如果您使用SoC平台（如SE、SM系列边缘设备），并使用它测试本例程，刷机后在`/opt/sophon/`下已经预装了相应的libsophon运行库包。

## 2. 编译程序

PCIE环境下和SOC环境下都可以直接进行编译。

进行编译之前，请先确认之前已经执行过`campplus/scripts/download.sh`

在开发板上或者X86主机执行如下编译：
您需要根据您使用的开发板及芯片种类进行选择

- 如果您是 `SoC平台` 请将参数设置为 `-DTARGET_ARCH=soc`；
- 如果您是 `x86 pcie BM1684X芯片` 请将参数设置为 `-DTARGET_ARCH=pcie`；
- 如果您是 `arm pcie BM1684X芯片`，由于arm平台之间差异较大，请参考[`kaldi编译`](#4-kaidi编译)部分

另外还需要安装第三方库，在Ubuntu系统或者Soc平台上请执行
```bash
sudo apt install libfst-dev libopenblas-dev
```

下面以BM1684X PCIE环境下的编译为例：

当前路径 `{campplus}/cpp`

```shell
mkdir build
cd build
cmake .. -DTARGET_ARCH=pcie
make -j4
```

## 3. 例程测试

在编译完成后，会在项目路径下生成campplus的可执行文件, model路径以及chip id都可以可以通过下列参数来指定，设置好以后即可运行

### 3.1 参数说明
```bash
usage: ./campplus [--model BMODEL] [--input INPUT_DIR] [--devid DEV_ID]
--model: 用于推理的bmodel路径；
--input: 用于存放wav音频文件的目录路径；
--devid: 用于推理的tpu设备id；
```

### 3.2 使用方式

将../dependencies/lib加入路径后运行`campplus`，如运行BM1684X fp32模型`campplus_bm1684x_fp32_1b.bmodel`:

```shell
export LD_LIBRARY_PATH=../dependencies/lib:$LD_LIBRARY_PATH
./campplus --bmodel=../models/BM1684X/campplus_bm1684x_fp32_1b.bmodel  --input=../datasets/test --dev_id=0
```

## 4. kaldi编译
如果在编译时遇到`kaldi`相关库的问题，可以自行从源码编译所需要的`kaldi`库，以pcie平台为例，具体步骤如下：

```bash
git clone https://github.com/kaldi-asr/kaldi
cd kaldi
mkdir build
cd build
cmake ..
cd src/feat
make -j
```

编译完成后，按照`campplus/cpp/dependencies/include`里所需要的头文件复制仓库中的头文件到对应目录，也可全部复制

```bash
cp -r kaldi/src/* campplus/cpp/dependencies/include/*
```

再根据`campplus/cpp/dependencies/lib_pcie_amd64`里所需要的库文件从`kaldi/build/src`对应目录复制到`campplus/cpp/dependencies/lib_pcie_amd64`中，所需要的库文件如下

```shell
kaldi/build/src/base/libkaldi-base.so
kaldi/build/src/feat/libkaldi-feat.so
kaldi/build/src/gmm/libkaldi-gmm.so
kaldi/build/src/matrix/libkaldi-matrix.so
kaldi/build/src/transform/libkaldi-transform.so
kaldi/build/src/tree/libkaldi-tree.so
kaldi/build/src/util/libkaldi-util.so
```
