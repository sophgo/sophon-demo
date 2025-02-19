# Python例程

## 目录

- [Python例程](#python例程)
  - [目录](#目录)
  - [1. 环境准备](#1-环境准备)
    - [1.1 x86/arm PCIe平台](#11-x86arm-pcie平台)
    - [1.2 SoC平台](#12-soc平台)
  - [2. 命令行推理测试](#2-命令行推理测试)
    - [2.1 参数说明](#21-参数说明)
  - [2. 推理测试](#2-推理测试)
    - [2.1 参数说明](#21-参数说明-1)
    - [2.2 使用方式](#22-使用方式)
  - [3. Web Demo](#3-web-demo)
    - [3.1 参数说明](#31-参数说明)
    - [3.2 程序流程图](#32-程序流程图)
    - [3.3 使用方式](#33-使用方式)

python目录下提供了一系列Python例程，具体情况如下：

| 序号 |  Python例程      |      说明           |
| ---- | --------------- | ------------------  |
| 1    | llama2.py       | Llama_sophon类的实现，使用SAIL进行命令行推理 |
| 2    | web_demo.py     | 支持多会话的web demo |

## 1. 环境准备

### 1.1 x86/arm PCIe平台

如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。

您还需要安装sophon-sail，现在可以直接通过dfss安装：

```bash
pip3 install dfss --upgrade
python3 -m dfss --install sail
```

此外您可能还需要安装其他第三方库：

```bash
pip3 install -r python/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 1.2 SoC平台

如果您使用SoC平台（如SE、SM系列边缘设备），在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包。若使用默认的python3.8.2环境，可直接安装编译好的sophon-sail包，指令如下：

```bash
pip3 install dfss --upgrade
python3 -m dfss --install sail
```

此外您可能还需要安装其他第三方库：

```bash
pip3 install -r python/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

如果想要自己编译sophon-sail的whl包，具体可参考[交叉编译安装sophon-sail](../../../docs/Environment_Install_Guide.md#42-交叉编译安装sophon-sail)。  

## 2. 命令行推理测试
python例程不需要编译，可以直接运行，PCIe平台和SoC平台的测试参数和运行方式是相同的。
### 2.1 参数说明

```bash
usage: llama2.py [--bmodel BMODEL] [--token TOKEN] [--dev_id DEV_ID]
--bmodel: 用于推理的bmodel路径；
--token: tokenizer的模型路径；
--dev_id: 用于推理的tpu设备id；
```
## 2. 推理测试
python例程不需要编译，可以直接运行，PCIe平台和SoC平台的测试参数和运行方式是相同的。
### 2.1 参数说明
llama2.py使用config/llama2.yaml配置文件进行参数配置。

llama2.yaml内容如下
```yaml
title: llama2-7b
bmodel_path: ../models/BM1684X/llama_w4f16_seq512_20250121_171104/llama_w4f16_seq512_20250121_171104.bmodel
token_path: ../models/BM1684X/llama_w4f16_seq512_20250121_171104/tokenizer
dev_ids: 0
```

### 2.2 使用方式

```bash
cd python
python3 llama2.py --config ./config/llama2.yaml
```
在读入模型后会显示"Question:"，然后输入就可以了。模型的回答会出现在"Answer"中。结束对话请输入"exit"。

## 3. Web Demo
我们提供了基于[streamlit](https://streamlit.io/)的web demo。
### 3.1 参数说明
web_demo.py使用config/llama2.yaml配置文件进行参数配置。

### 3.2 程序流程图

流程与Qwen例程基本相同，这里不重复赘述，可参考[Qwen](../../Qwen/python/README.md#32-程序流程图)

### 3.3 使用方式
通过streamlit运行web_demo.py即可运行一个web_server

```bash
python3 -m streamlit run web_demo.py -- --config=./config/web.yaml
```

首次运行需要输入邮箱，输入邮箱后命令行输出以下信息则表示启动成功
```bash
 You can now view your Streamlit app in your browser.

  Network URL: http://172.xx.xx.xx:8501
  External URL: http://103.xx.xxx.xxx:8501
```

在浏览器中打开输出的地址即可使用，在底部对话框中输入问题。