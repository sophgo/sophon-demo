# C++例程

## 目录

- [C++例程](#c例程)
  - [1. 环境准备](#1-环境准备)
    - [1.1 x86/arm/riscv PCIe平台](#11-x86armriscv-pcie平台)
    - [1.2 SoC平台](#12-soc平台)
  - [2. 程序编译](#2-程序编译)
    - [2.1 x86/arm/riscv PCIe平台](#21-x86armriscv-pcie平台)
    - [2.2 SoC平台](#22-soc平台)
  - [3. 推理测试](#3-推理测试)
    - [3.1 参数说明](#31-参数说明)
    - [3.2 测试音频](#32-测试音频)

cpp目录下提供了C++例程以供参考使用，具体情况如下：
| 序号 | C++例程        | 说明                                      |
| ---- | -------------- | ----------------------------------------  |
| 1    | silero_vad_bmrt | 使用BMRuntime裸接口推理，不依赖SAIL/OpenCV |

## 1. 环境准备
### 1.1 x86/arm/riscv PCIe平台
如果您在x86/arm/riscv平台安装了PCIe加速卡（如SC系列加速卡），可以直接使用它作为开发环境和运行环境。您需要安装libsophon，具体步骤可参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)或[riscv-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#6-riscv-pcie平台的开发和运行环境搭建)。

### 1.2 SoC平台
如果您使用SoC平台（如SE、SM系列边缘设备），刷机后在`/opt/sophon/`下已经预装了相应的libsophon运行库包，可直接使用它作为运行环境。通常还需要一台x86主机作为开发环境，用于交叉编译C++程序。

## 2. 程序编译
C++程序运行前需要编译可执行文件。

### 2.1 x86/arm/riscv PCIe平台
可以直接在PCIe平台上编译程序：

```bash
cd cpp/silero_vad_bmrt
mkdir build && cd build
cmake ..
make
cd ..
```
编译完成后，会在silero_vad_bmrt目录下生成silero_vad_bmrt.pcie。

### 2.2 SoC平台
通常在x86主机上交叉编译程序，您需要在x86主机上使用SOPHON SDK搭建交叉编译环境，将程序所依赖的头文件和库文件打包至soc-sdk目录中，具体请参考[交叉编译环境搭建](../../../docs/Environment_Install_Guide.md#41-交叉编译环境搭建)。本例程主要依赖libsophon运行库包。

交叉编译环境搭建好后，使用交叉编译工具链编译生成可执行文件：

```bash
cd cpp/silero_vad_bmrt
mkdir build && cd build
#请根据实际情况修改-DSDK的路径，需使用绝对路径。
cmake -DTARGET_ARCH=soc -DSDK=/path_to_sdk/soc-sdk ..
make
```
编译完成后，会在silero_vad_bmrt目录下生成silero_vad_bmrt.soc。

## 3. 推理测试
对于PCIe平台，可以直接在PCIe平台上推理测试；对于SoC平台，需将交叉编译生成的可执行文件及所需的模型、测试数据拷贝到SoC平台中测试。测试的参数及运行方式是一致的，下面主要以PCIe模式进行介绍。

### 3.1 参数说明
可执行程序默认有一套参数，请注意根据实际情况进行传参，`--input`参数为必填项。具体参数说明如下：

```bash
Usage: silero_vad_bmrt.pcie [options]

        --bmodel     <path>  bmodel file path (default: ../../models/BM1684X/silero_vad_bm1684x_f16.bmodel)
        --input      <path>  input WAV file path (required, 16kHz mono)
        --dev_id     <int>   TPU device id (default: 0)
        --threshold  <float> speech probability threshold, 0.0~1.0 (default: 0.5)
        --save_segments       save detected speech segments as separate WAV files
        --help                print this help
```

**注意：** C++传参与python不同，需要用`=`号指定参数值，例如`--bmodel=xxx.bmodel --input=test.wav`。`--save_segments`为bool参数，无需指定值。

### 3.2 测试音频
WAV音频测试实例如下：
```bash
cd cpp/silero_vad_bmrt
./silero_vad_bmrt.pcie --bmodel=../../models/BM1684X/silero_vad_bm1684x_f16.bmodel --input=../../datasets/test.wav --dev_id=0
```
测试结束后，输出VAD结果，内容包括：
- 检测到的语音段数量和每个段落的起止时间
- 每帧的平均推理耗时
- 实时因子（real_time_factor）

如需同时保存检测到的语音段为独立WAV文件，可加上`--save_segments`参数：
```bash
./silero_vad_bmrt.pcie --input=../../datasets/test.wav --save_segments
```

语音段输出示例：
```
Frames: 1875, speech segments: 19
  seg 0:    0.00s ->    2.05s (2.04s)
  seg 1:    2.63s ->    4.70s (2.07s)
...
```

结果会保存为JSON文件至`./results/`目录，语音段文件保存至`./results/segments/`目录。

### 3.3 性能测试
在SE7-32平台上测试的每帧平均性能如下：

| 阶段 | 耗时 (ms/frame) |
| ---- | -------------- |
| 音频读取 | 21.4 (一次性) |
| 预处理 (preprocess) | 0.003 |
| 推理 (inference) | 0.210 |
| 后处理 (postprocess) | 0.121 (一次性) |
| real_time_factor | 0.0066 |

> **说明**: 推理时间为纯TPU推理耗时（含S2D拷贝+D2S拷贝），即`inference`计时标签。预处理和后处理耗时极小，C++ bmrt裸接口的RTF比Python SAIL快约3.2倍。