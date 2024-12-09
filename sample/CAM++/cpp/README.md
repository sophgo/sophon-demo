# C++例程

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

## 1. 环境准备
### 1.1 x86/arm PCIe平台
如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。

- 请确保您的驱动及libsophon版本满足本例程的要求，具体请参考[简介](../README.md#1-简介)

### 1.2 SoC平台

如果您使用SoC平台（如SE、SM系列边缘设备），并使用它测试本例程，刷机后在`/opt/sophon/`下已经预装了相应的libsophon运行库包。

## 2. 编译程序

PCIE环境下和SOC环境下都需要直接执行如下编译：

当前路径 `{campplus}/cpp`

```shell
mkdir build
cd build
cmake ..
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
以SOC平台为例，如运行BM1684X fp32模型`campplus_bm1684x_fp32_1b.bmodel`:

```shell
./campplus --bmodel=../models/BM1684X/campplus_bm1684x_fp32_1b.bmodel  --input=../datasets/test --dev_id=0
```
