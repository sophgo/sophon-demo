# Python例程

## 目录
- [1. 环境准备](#1-环境准备)
  - [1.1 x86/arm/riscv PCIe平台](#11-x86armriscv-pcie平台)
  - [1.2 SoC平台](#12-soc平台)
- [2. 推理测试](#2-推理测试)
  - [2.1 参数说明](#21-参数说明)
  - [2.2 测试说话人识别](#22-测试说话人识别)

python目录下提供了campplus.py文件，其使用kaldi前处理、SAIL推理。

## 1. 环境准备
### 1.1 x86/arm/riscv PCIe平台

如果您在x86/arm/riscv平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon、sophon-opencv、sophon-ffmpeg和sophon-sail，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)或[riscv-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#6-riscv-pcie平台的开发和运行环境搭建)。

此外您可能还需要安装其他第三方库：
```bash
pip3 install torchaudio
```
注：torchaudio和torch的版本需要相同，如果出现torchaudio的报错，请检查torch和torchaudio的版本。

### 1.2 SoC平台

如果您使用SoC平台（如SE、SM系列边缘设备），并使用它测试本例程，刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包。您还需要交叉编译安装sophon-sail，具体可参考[交叉编译安装sophon-sail](../../../docs/Environment_Install_Guide.md#42-交叉编译安装sophon-sail)。

## 2. 推理测试
python例程不需要编译，可以直接运行，PCIe平台和SoC平台的测试参数和运行方式是相同的。
### 2.1 参数说明
```bash
usage: campplus.py [--input INPUT_PATH] [--bmodel BMODEL] [--dev_id DEV_ID]
--input: 存放wav音频文件的路径；
--bmodel: 用于推理的bmodel路径，默认使用stage 0的网络进行推理；
--dev_id: 用于推理的tpu设备id；
```

### 2.2 测试说话人识别
测试实例如下，对三段简短语音计算embedding并将结果存储在results文件夹中
```bash
python3 python/campplus.py --dev_id 0 --bmodel models/BM1684X/campplus_bm1684x_fp32_1b.bmodel --input datasets/test
```
测试结束后，会打印推理时间等信息，并将计算结果保存在`./results`中。
