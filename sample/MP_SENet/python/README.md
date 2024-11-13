# Python例程

## 目录

* [1. 环境准备](#1-环境准备)
    * [1.1 x86 PCIe平台](#11-x86-pcie平台)
    * [1.2 SoC平台](#12-soc平台)
* [2. 推理测试](#2-推理测试)
    * [2.1 参数说明](#21-参数说明)
    * [2.2 使用方式](#22-使用方式)

## 1. 环境准备
### 1.1 x86 PCIe平台

如果您在x86平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon以及sophon-mw，除此之外，您还需要编译并安装sophon-sail，以上安装步骤具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)。

此外您还需要安装其他第三方库，在本例程顶层目录MP_SENet/执行
```bash
pip3 install -r python/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```


### 1.2 SoC平台

如果您使用SoC平台（如SE、SM系列边缘设备），并使用它测试本例程，刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包。

此外您还需要安装其他第三方库：
```bash
pip3 install -r python/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

您还需要交叉编译安装sophon-sail，具体可参考[交叉编译安装sophon-sail](../../../docs/Environment_Install_Guide.md#42-交叉编译安装sophon-sail)。


## 2. 推理测试
python例程不需要编译，可以直接运行，PCIe平台和SoC平台的测试参数和运行方式是相同的。
### 2.1 参数说明

```bash
usage: mp_senet_sail.py [--mp_senet_model MP_SENet_BMODEL] [--wav_files *.wav] [--result_files RESULT_FILES] [--dev_id DEV_ID] [--compress_factor COMPRESS_FACTOR] [--sampling_rate SAMPLING_RETE] [--n_fft N_FFT] [--hop_size HOP_SIZE] [--win_size WIN_SIZE]
--mp_senet_model: 用于mp_senet_model推理的bmodel路径；
--wav_files: 用于推理的噪声音频文件路径；
--result_files: 用于存储推理后生成的去噪音频路径；
--dev_id: 用于推理的tpu设备id；
--compress_factor: 用于调整幅度谱压缩程度的压缩因子，优化信号处理效果；
--sampling_rate: 用于指定处理音频的采样率；
--n_fft: 用于指定执行快速傅里叶变换（FFT）的点数；
--hop_size: 用于指定连续窗口开始之间的样本数；
--win_size: 用于指定窗口大小；

```

### 2.2 使用方式
- 准备音频数据

您可以运行下载脚本(scripts/download.sh)获取测试数据。您也可以自行上传wav数据至‘datasets/’目录下。
- 运行例程

在本例程顶层目录MP_SENet/执行（其余参数使用默认设置）：
```bash
python3 python/mp_senet_sail.py --mp_senet_model ./models/BM1684X/mpsenet_vb_1b_bf16.bmodel --wav_files  ./datasets/ --result_files ./python/results --dev_id 0
```
测试结束后，会将推理得到的音频文件保存在‘python/results/’下

## 3. 流程图
![alt text](process.png)