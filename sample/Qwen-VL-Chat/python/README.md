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

由于Qwen-vl需要上传/切换图片，python目录下提供了web_demo例程，具体情况如下：

| 序号 |  Python例程       |            说明                 |
| ---- | ---------------- | ------------------------------ |
|   1  | web_demo.py      | 使用SAIL推理，支持多会话的web demo |

## 1. 环境准备

### 1.1 x86/arm PCIe平台

如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon和sophon-sail，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。

此外您可能还需要安装其他第三方库：

```bash
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 1.2 SoC平台

如果您使用SoC平台（如SE、SM系列边缘设备），并使用它测试本例程，请使用V24.04.01刷机包进行刷机，刷机成功后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包。您还需要交叉编译安装sophon-sail，具体可参考[交叉编译安装sophon-sail](../../../docs/Environment_Install_Guide.md#42-交叉编译安装sophon-sail)。  

若使用刷机后默认的python3.8.2环境，可直接安装编译好的sophon-sail包，指令如下：

```bash
pip3 install dfss --upgrade
python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen-VL/sophon_arm-3.8.0-py3-none-any.whl
pip3 install sophon_arm-3.8.0-py3-none-any.whl --force-reinstall
```

此外您可能还需要安装其他第三方库：

```bash
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 2. 推理测试

python例程不需要编译，可以直接运行，PCIe平台和SoC平台的测试参数和运行方式是相同的。

### 2.1 参数说明

本例程的参数配置在`python/config.ini`文件中。

```bash
usage: config.ini [--bmodel BMODEL] [--token TOKEN] [--dev_id DEV_ID]
--bmodel: 用于推理的bmodel路径；
--token: tokenizer目录路径；
--dev_id: 用于推理的tpu设备id；
```

### 2.2 使用方式

通过[streamlit](https://streamlit.io/)启动web server。

```bash
streamlit run python/web_demo.py
```

根据终端打印的url信息，在浏览器中访问相应的url地址，即可开始使用。
