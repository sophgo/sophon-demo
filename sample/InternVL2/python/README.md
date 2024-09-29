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
|   1  | internv2_sail.py | 使用SAIL推理的例程 |

## 1. 环境准备

### 1.1 x86/arm PCIe平台

如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon和sophon-sail，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。

此外您可能还需要安装其他第三方库：

```bash
pip3 install -r python/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

您还需要安装sophon-sail，由于本例程需要的sophon-sail版本较新，可以用如下命令下载sophon-sail源码，并参考[sophon-sail python3接口编译安装指南](https://doc.sophgo.com/sdk-docs/v24.04.01/docs_latest_release/docs/sophon-sail/docs/zh/html/1_build.html#python3wheel)自己编译sophon-sail，该例程编译不包含bmcv,sophon-ffmpeg,sophon-opencv的SAIL的即可。

```bash
pip3 install dfss --upgrade
python3 -m dfss --url=open@sophgo.com:sophon-demo/InternVL2/sophon-sail_3.9.0.tar.gz
tar xvf sophon-sail.tar.gz
```

### 1.2 SoC平台

如果您使用SoC平台（如SE、SM系列边缘设备），在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包。若使用默认的python3.8.2环境，可直接安装编译好的sophon-sail包，指令如下：

```bash
pip3 install dfss --upgrade

#如果是在bm1684x环境上，下载这个：
python3 -m dfss --url=open@sophgo.com:sophon-demo/InternVL2/whls/se7/sophon_arm-3.9.0-py3-none-any.whl

#如果是在bm1688环境上，下载这个：
python3 -m dfss --url=open@sophgo.com:sophon-demo/InternVL2/whls/se9/sophon_arm-3.8.0-py3-none-any.whl

pip3 install sophon_arm-3.8.0-py3-none-any.whl --force-reinstall
```

此外您可能还需要安装其他第三方库：

```bash
pip3 install -r python/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

如果想要自己编译sophon-sail的whl包，具体可参考[交叉编译安装sophon-sail](../../../docs/Environment_Install_Guide.md#42-交叉编译安装sophon-sail)。  

## 2. 推理测试

python例程不需要编译，可以直接运行，PCIe平台和SoC平台的测试参数和运行方式是相同的。

### 2.1 参数说明

```bash
usage: internvl2_sail.py [-h] -m MODEL_PATH [-t TOKENIZER] [-d DEVID]

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_PATH, --model_path MODEL_PATH
                        path to the bmodel file
  -t TOKENIZER, --tokenizer TOKENIZER
                        path to the tokenizer file
  -d DEVID, --devid DEVID
                        device ID to use
```

### 2.2 使用方式

```bash
cd python
python3 internvl2_sail.py --model_path internvl2-4b_bm1684x_int4.bmodel --tokenizer token_config_4b --devid 0
```

注意，如果跑4b模型，使用4b的tokenizer。同理，2b模型使用2b的tokenizer。

效果图：
![Alt text](../pics/image.png)