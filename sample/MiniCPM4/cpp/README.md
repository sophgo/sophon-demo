[简体中文](./README.md)

# C++例程
cpp目录下提供了C++例程以供参考使用，具体情况如下：
| 序号  | C++例程      | 说明                                 |
| ---- | ------------- | -----------------------------------  |
| 1    | minicpm4     |         使用BMRT推理                 |

## 目录

- [C++例程](#c例程)
  - [目录](#目录)
  - [1. 环境准备](#1-环境准备)
    - [1.1 x86/arm/riscv PCIe平台](#11-x86armriscv-pcie平台)
    - [1.2 SoC平台](#12-soc平台)
  - [2. 程序编译](#2-程序编译)
    - [2.1 下载第三方库](#21-下载第三方库)
    - [2.2 编译](#22-编译)
  - [3. 推理测试](#3-推理测试)
    - [3.1 参数说明](#31-参数说明)

## 1. 环境准备
### 1.1 x86/arm/riscv PCIe平台
如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），可以直接使用它作为开发环境和运行环境。您需要安装libsophon，具体步骤可参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)或[riscv-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#6-riscv-pcie平台的开发和运行环境搭建)。

### 1.2 SoC平台
如果您使用SoC平台（如SE、SM系列边缘设备），刷机后在`/opt/sophon/`下已经预装了相应的libsophon运行库包，可直接使用它作为运行环境。


## 2. 程序编译
C++程序运行前，需要先编译为可执行文件。PCIe和SoC平台方法相同，均可直接编译。
### 2.1 下载第三方库
在cpp目录下执行以下操作
```bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen/cpp/third_party.zip
unzip third_party.zip
rm -f third_party.zip
```
### 2.2 编译
在cpp目录下执行以下操作
```bash
mkdir build && cd build
cmake ..
make
```
编译完成后，会在build目录下生成minicpm4。


## 3. 推理测试
### 3.1 参数说明
请注意根据实际情况进行传参：
```bash
Usage: minicpm4 [params] 

  --help            : Show help info.
  --model           : Set model path.
  --tokenizer       : Set tokenizer path, if not provided, use ../../python/token_config/tokenizer.json
  --devid           : Set devices to run for model, e.g. 1,2, if not provided, use 0
```
> **注意：** cpp例程传参与python不同，需要用等于号，以下为BM1684X的示例：
```bash
./minicpm4 --model=../../models//BM1684X/minicpm4-8b_w4bf16_seq512_bm1684x_1dev_20250613_175044.bmodel --tokenizer=../../python/token_config/tokenizer.json
```
模型下载或者编译请参考[模型下载或者编译](../README.md)。
