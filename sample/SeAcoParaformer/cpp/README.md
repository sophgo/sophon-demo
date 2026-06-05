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
    * [3.2 测试音频](#32-测试音频)

cpp目录下提供了一系列C++例程以供参考使用，具体情况如下：
| 序号 | C++例程                | 说明                                        |
| ---- | ---------------------- | ------------------------------------------- |
| 1    | seaco_paraformer_bmrt  | 使用libsndfile解码、bmrt推理、Armadillo预处理 |


## 1. 环境准备
### 1.1 x86/arm/riscv PCIe平台
如果您在x86/arm/riscv平台安装了PCIe加速卡（如SC系列加速卡），可以直接使用它作为开发环境和运行环境。您需要安装libsophon、libsndfile和Armadillo，具体步骤可参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)或[riscv-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#6-riscv-pcie平台的开发和运行环境搭建)。

此外还需要安装以下依赖：
```bash
sudo apt-get install -y libsndfile1-dev libarmadillo-dev
```

### 1.2 SoC平台
如果您使用SoC平台（如SE、SM系列边缘设备），刷机后在`/opt/sophon/`下已经预装了相应的libsophon运行库包，可直接使用它作为运行环境。通常还需要一台x86主机作为开发环境，用于交叉编译C++程序。

需要在x86主机上安装交叉编译依赖，请运行`scripts/download.sh`下载cross_compile_module和3rd_party依赖库：

```bash
cd scripts
./download.sh
```

## 2. 程序编译
### 2.1 x86/arm/riscv PCIe平台
可以直接在PCIe平台上编译程序：

```bash
cd cpp/seaco_paraformer_bmrt
mkdir build && cd build
cmake .. && make
cd ../..
```
编译完成后，会在seaco_paraformer_bmrt目录下生成seaco_paraformer_bmrt.pcie。

### 2.2 SoC平台
通常在x86主机上交叉编译程序，您需要在x86主机上使用SOPHON SDK搭建交叉编译环境，将程序所依赖的头文件和库文件打包至soc-sdk目录中，具体请参考[交叉编译环境搭建](../../../docs/Environment_Install_Guide.md#41-交叉编译环境搭建)。本例程主要依赖libsophon运行库包，以及cross_compile_module中的3rd_party库（libsndfile、Armadillo、BLAS、LAPACK）。

交叉编译环境搭建好后，使用交叉编译工具链编译生成可执行文件：

```bash
cd cpp/seaco_paraformer_bmrt
mkdir build && cd build
#请根据实际情况修改-DSDK的路径，需使用绝对路径。
cmake -DTARGET_ARCH=soc -DSDK=/path_to_sdk/soc-sdk ..
make
cd ../..
```
编译完成后，会在seaco_paraformer_bmrt目录下生成seaco_paraformer_bmrt.soc。

## 3. 推理测试
对于PCIe平台，可以直接在PCIe平台上推理测试；对于SoC平台，需将交叉编译生成的可执行文件及所需的模型、测试数据拷贝到SoC平台中测试。测试的参数及运行方式是一致的，下面主要以PCIe模式进行介绍。

### 3.1 参数说明
可执行程序默认有一套参数，请注意根据实际情况进行传参。具体参数说明如下：

```bash
usage:./seaco_paraformer_bmrt.pcie [params]

        --model_dir (value:../../models/BM1684X)
                model directory path
        --dev_id (value:0)
                TPU device id
        --input (value:../../audio/asr_example.wav)
                input audio file path (.wav format, 16kHz, mono)
```
**注意：** CPP传参与python不同，需要用等于号，例如`./seaco_paraformer_bmrt.pcie --bmodel=xxx`。

### 3.2 测试音频
音频测试实例如下：
```bash
# 单文件推理测试
./seaco_paraformer_bmrt.pcie \
    --model_dir=../../models/BM1684X \
    --input=../../audio/asr_example.wav \
    --dev_id=0
```
测试结束后，会打印识别结果文本、预处理耗时、encoder耗时、decoder耗时、总耗时和RTF（实时率）等信息。
