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

Janus能够输入单一图片连续对话，python目录下提供了例程，具体情况如下：

| 序号  |  Python例程       |            说明                 |
| ---- | ---------------- | ------------------------------ |
|   1  | janus.py      | 使用SAIL推理                     |

## 1. 环境准备

### 1.1 x86/arm PCIe平台

- 需要**SDK v24.04.01及其以上版本**

- 如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon、sophon-opencv、sophon-ffmpeg，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。

- 此外您可能还需要安装其他第三方库：

```bash
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

- 本例程依赖sophon-sail，可直接安装编译好的sophon-sail包，执行如下命令：

```bash
pip3 install dfss --upgrade
python3 -m dfss --install sail
```

- 需要下载运行配置文件，执行如下命令

```bash
cd python
python3 -m dfss --url=open@sophgo.com:sophon-demo/Janus/processor_config.zip
unzip processor_config.zip
rm processor_config.zip
```

### 1.2 SoC平台

- 需要**SDK v24.04.01及其以上版本**

- 如果您使用SoC平台（如SE、SM系列边缘设备），并使用它测试本例程，请使用**SDK V24.04.01及其以上版本**对应的刷机包进行刷机，刷机成功后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包。

- 此外您可能还需要安装其他第三方库：

```bash
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
``` 

- 本例程依赖sophon-sail，可直接安装编译好的sophon-sail包，执行如下命令：

```bash
pip3 install dfss --upgrade
python3 -m dfss --install sail
```

- 需要下载运行配置文件，执行如下命令

```bash
cd python
python3 -m dfss --url=open@sophgo.com:sophon-demo/Janus/processor_config.zip
unzip processor_config.zip
rm processor_config.zip
```

## 2. 推理测试

python例程不需要编译，可以直接运行，PCIe平台和SoC平台的测试参数和运行方式是相同的。

### 2.1 参数说明
janus.py使用config/janus.yaml配置文件进行参数配置。

janus.yaml内容如下
```yaml
bmodel_path: ../models/BM1684X/janus-pro-7b_int4_seq2048.bmodel   ## 用于推理的bmodel路径
token_path: ./processor_config    ## tokenizer目录路径；
dev_ids: 0   ## 用于推理的tpu设备id；
image_path: ../pics/test.jpg  ##用于推理的图片路径
```

### 2.2 使用方式


- 为了测试图片，可以参考执行如下命令
```bash
python3 janus.py --config ./config/janus.yaml
```

在Question: 处进行提问，例如：what's in the room?
终端将打印FTL、TPS性能数据，并输出回答结果，接着可进一步对视频进行提问，输入q即可退出。如果您想重新更换照片，您可以在终端输入clear或者new即可，然后按照提示输入新的图片地址。
