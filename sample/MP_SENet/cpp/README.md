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

cpp目录下提供C++例程以供参考使用，说明如下：
| 序号  | C++例程  | 说明                                                        |
| ---- | ------- | ----------------------------------------------------------- |
| 1    | mp_senet | 使用bmcv前处理（stft）、BMRT推理、bmcv后处理（istft），音频降噪 |

> 与Python例程类似，C++例程需要预先准备好bmodel，运行时加载。模型准备方式见[准备模型与数据](../README.md#3-数据准备与模型编译)。


## 1. 环境准备
### 1.1 x86/arm/riscv PCIe平台
如果您在x86/arm/riscv平台安装了PCIe加速卡（如SC系列加速卡），可以直接使用它作为开发环境和运行环境。您需要安装libsophon、sophon-opencv和sophon-ffmpeg，具体步骤可参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)或[riscv-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#6-riscv-pcie平台的开发和运行环境搭建)。

### 1.2 SoC平台
如果您使用SoC平台（如SE、SM系列边缘设备），刷机后在`/opt/sophon/`下已经预装了相应的libsophon运行库包，可直接使用它作为运行环境。通常还需要一台x86主机作为开发环境，用于交叉编译C++程序。

## 2. 程序编译
### 2.1 x86/arm/riscv PCIe平台
可以直接在PCIe平台上编译程序：

```bash
cd mp_senet
mkdir build && cd build
cmake .. && make # 生成mp_senet.pcie
cd ..
```
编译完成后，会在mp_senet目录下生成mp_senet.pcie。

### 2.2 SoC平台
通常在x86主机上交叉编译程序，您需要在x86主机上使用SOPHON SDK搭建交叉编译环境，将程序所依赖的头文件和库文件打包至soc-sdk目录中，具体请参考[交叉编译环境搭建](../../../docs/Environment_Install_Guide.md#41-交叉编译环境搭建)。本例程主要依赖libsophon运行库包。

交叉编译环境搭建好后，使用交叉编译工具链编译生成可执行文件：

```bash
cd cpp/mp_senet
mkdir build && cd build
#请根据实际情况修改-DSDK的路径，需使用绝对路径。
cmake -DTARGET_ARCH=soc -DSDK=/path_to_sdk/soc-sdk ..
make
cd ..
```
编译完成后，会在mp_senet目录下生成mp_senet.soc。

SoC平台也可以直接在板子上原生编译：

```bash
cd cpp/mp_senet
mkdir build && cd build
cmake -DTARGET_ARCH=soc -DSDK=/opt/sophon/libsophon-current .. && make
cd ..
```

## 3. 推理测试
对于PCIe平台，可以直接在PCIe平台上推理测试；对于SoC平台，需将编译生成的可执行文件及所需的模型、测试数据拷贝到SoC平台中测试。测试的参数及运行方式是一致的，下面主要以SoC模式进行介绍。

### 3.1 参数说明
可执行程序默认有一套参数，请注意根据实际情况进行传参，具体参数说明如下：

```bash
usage:./mp_senet.soc [params]

        --bmodel (value:../models/BM1684X/mpsenet_vb_1b_bf16.bmodel)
                bmodel file path
        --wav (value:../datasets/test.wav)
                input WAV file path
        --output (value:./results/test_denoised.wav)
                output WAV file path
        --dev_id (value:0)
                TPU device id
        --compress_factor (value:0.27)
                compress factor (bf16=0.27, fp32=0.30)
        --sampling_rate (value:16000)
                sampling rate
        --n_fft (value:400)
                n_fft
        --hop_size (value:150)
                hop size (bf16=150, fp32=100)
        --win_size (value:400)
                window size
        --help
                print help information.
```
**注意：** 程序会根据bmodel文件名自动判断bf16/fp32模式，并设置对应的compress_factor和hop_size（bf16: 0.27/150，fp32: 0.30/100），也可通过命令行参数手动覆盖。

### 3.2 测试音频
测试实例如下：

```bash
./mp_senet.soc --bmodel=../models/BM1684X/mpsenet_vb_1b_bf16.bmodel --wav=../datasets/test.wav --output=results/test_denoised.wav --dev_id=0
```

运行后会在指定路径生成降噪后的WAV文件，并打印推理耗时统计信息：

```
======================================
MP_SENet C++ Demo
  bmodel:     ../models/BM1684X/mpsenet_vb_1b_bf16.bmodel
  input wav:  ../datasets/test.wav
  output wav: results/test_denoised.wav
  dev_id:     0
  mode:       bf16
  compress:   0.27
  hop_size:   150
======================================
Loaded bmodel: ../models/BM1684X/mpsenet_vb_1b_bf16.bmodel
Network: mpsenet  max_length=640  max_wav_length=95999  compress_factor=0.27  hop_size=150
Input: ../datasets/test.wav  sr=48000  samples=576000
Resampled 48000 → 16000Hz, 192000 samples
2 chunks, stft T=640
Wrote: results/test_denoised.wav (191998 samples)

############################
SUMMARY: MP_SENet
############################
[ mp_senet inference ]  loops:    2 avg: 1397.35 ms
```

与Python例程结果对比（测试音频为12秒带噪语音，bf16模型）：

| 测试平台 | 测试程序 | 模型 | 前处理+推理+后处理总时间 |
| ------- | ------- | ---- | ---------------------- |
| SE7-32  | mp_senet_sail.py | mpsenet_vb_1b_bf16.bmodel | ~3103ms |
| SE7-32  | mp_senet (C++)   | mpsenet_vb_1b_bf16.bmodel | ~2800ms |
