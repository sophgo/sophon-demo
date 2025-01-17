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

Llama3_2_Vision能够输入单一图片连续对话，python目录下提供了例程，具体情况如下：

| 序号  |  Python例程       |            说明                 |
| ---- | ---------------- | ------------------------------ |
|   1  | llama3.py      | 使用SAIL推理                     |

## 1. 环境准备

### 1.1 x86/arm PCIe平台

- 需要**SDK v24.04.01及其以上版本**

- 如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon、sophon-opencv、sophon-ffmpeg，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。

- 此外您可能还需要安装其他第三方库：

```bash
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

- 您还需要安装sophon-sail，由于本例程需要的sophon-sail版本较新，可以用如下命令下载sophon-sail源码，并参考[sophon-sail python3接口编译安装指南](https://doc.sophgo.com/sdk-docs/v24.04.01/docs_latest_release/docs/sophon-sail/docs/zh/html/1_build.html#python3wheel)自行编译sophon-sail。

```bash
pip3 install dfss --upgrade
python3 -m dfss --url=open@sophgo.com:sophon-demo/Llama3_2_Vision/sophon-sail.zip
unzip sophon-sail.zip
```

- 需要下载运行配置文件，执行如下命令

```bash
cd python
python3 -m dfss --url=open@sophgo.com:sophon-demo/Llama3_2_Vision/token_configs.zip
unzip token_configs.zip
rm token_configs.zip
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
python3 -m dfss --url=open@sophgo.com:sophon-demo/Llama3_2_Vision/sophon_arm-3.9.0-py3-none-any.whl
pip3 install sophon_arm-3.9.0-py3-none-any.whl --force-reinstall
```

- 您还可以自行交叉编译安装sophon-sail，由于本例程需要的sophon-sail版本较新，可以先用如下命令下载sophon-sail源码，接着可参考[交叉编译安装sophon-sail](../../../docs/Environment_Install_Guide.md#42-交叉编译安装sophon-sail)。 

```bash
pip3 install dfss --upgrade
python3 -m dfss --url=open@sophgo.com:sophon-demo/Llama3_2_Vision/sophon-sail.zip
unzip sophon-sail.zip
```

- 需要下载运行配置文件，执行如下命令

```bash
cd python
python3 -m dfss --url=open@sophgo.com:sophon-demo/Llama3_2_Vision/token_configs.zip
unzip token_configs.zip
rm token_configs.zip
```

## 2. 推理测试

python例程不需要编译，可以直接运行，PCIe平台和SoC平台的测试参数和运行方式是相同的。

### 2.1 参数说明
llama3.py使用config/llama3.yaml配置文件进行参数配置。

llama3.yaml内容如下
```yaml
bmodel_path: ../models/BM1684X/llama3.2-11b-vision_int4_512seq.bmodel   ## 用于推理的bmodel路径
token_path: ./token_config    ## tokenizer目录路径；
dev_ids: 0   ## 用于推理的tpu设备id；
image_path: ../pics/test.jpg  ##用于推理的图片路径
```

### 2.2 使用方式


- 为了测试图片，可以参考执行如下命令
```bash
python3 llama3.py --config ./config/llama3.yaml
```

在Question: 处进行提问，例如：what's in the room?
终端将打印FTL、TPS性能数据，并输出回答结果，接着可进一步对视频进行提问，输入q即可退出。如果您想重新更换照片，您可以在终端输入clear或者new即可，然后按照提示输入新的图片地址。
