[简体中文](./README.md)

# C++例程

## 目录

- [C++例程](#c例程)
  - [目录](#目录)
  - [1. 环境准备](#1-环境准备)
    - [1.1 x86/arm/riscv PCIe平台](#11-x86armriscv-pcie平台)
    - [1.2 SoC平台](#12-soc平台)
  - [2. 程序编译](#2-程序编译)
  - [3. 推理测试](#3-推理测试)
    - [3.1 参数说明](#31-参数说明)
  - [4. Tokenizer编译方法](#4-tokenizer编译方法)


## 1. 环境准备
### 1.1 x86/arm/riscv PCIe平台
如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），可以直接使用它作为开发环境和运行环境。您需要安装libsophon，具体步骤可参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)或[riscv-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#6-riscv-pcie平台的开发和运行环境搭建)。

### 1.2 SoC平台
如果您使用SoC平台（如SE、SM系列边缘设备），刷机后在`/opt/sophon/`下已经预装了相应的libsophon运行库包，可直接使用它作为运行环境。通常还需要一台x86主机作为开发环境，用于交叉编译C++程序。


## 2. 程序编译
C++程序运行前需要编译可执行文件，您无需交叉编译，完成环境准备后，您可以在x86/aarch64等架构的平台上直接编译可执行程序。
```bash
cd cpp
mkdir build && cd build
cmake ..
make
cd ..
```
编译完成后，会在build目录下生成./pipeline。

## 3. 推理测试
对于PCIe平台，可以直接在PCIe平台上推理测试；对于SoC平台，需将交叉编译生成的可执行文件及所需的模型、测试数据拷贝到SoC平台中测试。测试的参数及运行方式是一致的，下面主要以PCIe模式进行介绍。

### 3.1 参数说明
可执行程序默认有一套参数，请注意根据实际情况进行传参，以qwen_bmlib.pcie为例，具体参数说明如下：
```bash
Usage:
  -h, --help        : Show help info 
  -m, --model       : Set model path 
  -c, --config      : Set config path 
  -s, --do_sample   : Enable sampling during generation
  -d, --devid       : Set devices to run for model, default is '0'

### 3.2 测试

需要在`cpp/build`目录下执行程序，测试实例如下。
```bash
 ./pipeline --model ../../models/paddleocr-vl_bf16_seq2048_bm1688_1core_static_20260221_195626.bmodel --config ../../config/ --devid 0
```
运行此命令，进入对话后，需要输入mode、image path，然后模型会给出回答。

## 4. Tokenizer编译方法

本例程的下载脚本提供了编译好的tokenizer，如果您需要自己编译，可以参考以下步骤：

- 下载源码
```bash
git clone https://github.com/mlc-ai/tokenizers-cpp
cd tokenizers-cpp
```

- 升级RUST编译工具包
```bash
export RUSTUP_DIST_SERVER="https://rsproxy.cn"
export RUSTUP_UPDATE_ROOT="https://rsproxy.cn/rustup"
curl --proto '=https' --tlsv1.2 -sSf https://rsproxy.cn/rustup-init.sh | sh
```

- 直接在目标平台编译
```bash
mkdir build && cd build
cmake ..
make
```

- 拷贝静态库和头文件`build/libtokenizers_c.a`、`build/libtokenizers_cpp.a`、`build/sentencepiece/libsentencepiece.a`、`build/release/libtokenizers_c.a`和`/include`到工作目录，即可