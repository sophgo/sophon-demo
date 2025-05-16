# Python例程

## 目录

- [Python例程](#python例程)
  - [目录](#目录)
  - [1. 环境准备](#1-环境准备)
    - [1.1 x86/arm PCIe平台](#11-x86arm-pcie平台)
    - [1.2 SoC平台](#12-soc平台)
  - [2. 推理测试](#2-推理测试)
    - [2.1 参数说明](#21-参数说明)
    - [2.2 使用方式](#22-使用方式)

| 序号  |  Python例程       |            说明                 |
| ---- | ---------------- | ------------------------------ |
|   1  | phi4mm.py | 使用SAIL推理的例程 |

## 1. 环境准备

### 1.1 x86/arm PCIe平台

如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。

您还需要安装sophon-sail，x86 pcie平台可以直接通过dfss安装：

```bash
pip3 install dfss --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple
python3 -m dfss --install sail
```

arm等其他pcie平台需要下载sophon-sail源码包：
```bash
python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv8_plus_seg_fuse/sophon-sail.tar.gz
```
下载完成后，参考[sophon-sail编译安装指南](https://doc.sophgo.com/sdk-docs/v24.04.01/docs_latest_release/docs/sophon-sail/docs/zh/html/1_build.html#)编译。

此外您可能还需要安装其他第三方库：

```bash
pip3 install -r python/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 1.2 SoC平台

如果您使用SoC平台（如SE、SM系列边缘设备），在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包。若使用默认的python3.8.2环境，SoC平台可直接安装编译好的sophon-sail包，指令如下：

```bash
pip3 install dfss --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple
python3 -m dfss --install sail
```

此外您可能还需要安装其他第三方库：

```bash
pip3 install -r python/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 2. 推理测试

python例程不需要编译，可以直接运行，PCIe平台和SoC平台的测试参数和运行方式是相同的。

### 2.1 参数说明

```bash
usage: phi4mm.py [-h] -m MODEL_PATH [-t PROCESSOR] [-d DEVID]

options:
  -h, --help            show this help message and exit
  -m MODEL_PATH, --model_path MODEL_PATH
                        path to the bmodel file
  -t PROCESSOR, --processor PROCESSOR
                        path to the tokenizer file
  -d DEVID, --devid DEVID
                        device ID to use
```

### 2.2 使用方式

```bash
cd python
python3 phi4mm.py --model_path phi4mm_bm1684x_int4_1core.bmodel --processor ./processor/ --devid 0
```