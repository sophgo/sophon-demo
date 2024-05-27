# Python例程

## 目录

* [1. 环境准备](#1-环境准备)
    * [1.1 x86/arm PCIe平台](#11-x86arm-pcie平台)
    * [1.2 SoC平台](#12-soc平台)
* [2. 命令行推理测试](#2-命令行推理测试)
    * [2.1 参数说明](#21-参数说明)
    * [2.2 使用方式](#22-使用方式)
* [3. 支持多会话的Web Demo](#3-支持多会话的Web-Demo)
    * [3.1 参数说明](#31-参数说明)
    * [3.2 使用方式](#32-使用方式)

python目录下提供了一系列Python例程，具体情况如下：

| 序号 |  Python例程      |      说明           |
| ---- | --------------- | ------------------  |
| 1    | llama2.py       | Llama_sophon类的实现，使用SAIL进行命令行推理 |
| 2    | web_demo.py     | 支持多会话的web demo |


## 1. 环境准备
### 1.1 x86/arm PCIe平台

如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon、sophon-opencv、sophon-ffmpeg，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。

此外您还需要安装其他第三方库：
```bash
pip3 install -r python/requirements.txt
```
您还需要安装sophon-sail，由于本例程需要的sophon-sail版本较新，相关功能还未发布，这里暂时提供一个可用的sophon-sail版本，可以通过以下命令下载源码，参考[sophon-sail编译安装指南](https://doc.sophgo.com/sdk-docs/v23.09.01-lts/docs_latest_release/docs/sophon-sail/docs/zh/html/1_build.html#id11#)自己编译sophon-sail。
```bash
python3 -m dfss --url=open@sophgo.com:sophon-demo/Llama2/sail/sophon-sail_20240417.tar.gz
tar xvf sophon-sail_20240417.tar.gz
```
### 1.2 SoC平台

如果您使用SoC平台（如SE、SM系列边缘设备），并使用它测试本例程，刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包。

此外您还需要安装其他第三方库：
```bash
pip3 install -r python/requirements.txt
```
您还需要安装sophon-sail，由于本例程需要的sophon-sail版本较新，相关功能还未发布，这里暂时提供一个可用的sophon-sail版本，可以通过以下命令下载源码，参考[sophon-sail编译安装指南](https://doc.sophgo.com/sdk-docs/v23.09.01-lts/docs_latest_release/docs/sophon-sail/docs/zh/html/1_build.html#id11#)自己编译sophon-sail。
```bash
python3 -m dfss --url=open@sophgo.com:sophon-demo/Llama2/sail/sophon-sail_20240417.tar.gz
tar xvf sophon-sail_20240417.tar.gz
```
## 2. 命令行推理测试
python例程不需要编译，可以直接运行，PCIe平台和SoC平台的测试参数和运行方式是相同的。
### 2.1 参数说明

```bash
usage: llama2.py [--bmodel BMODEL] [--token TOKEN] [--dev_id DEV_ID]
--bmodel: 用于推理的bmodel路径；
--token: tokenizer的模型路径；
--dev_id: 用于推理的tpu设备id；
```

### 2.2 使用方式

```bash
python3 python/llama2.py --bmodel models/BM1684X/llama2-7b_int4_1dev.bmodel --token python/token_config/tokenizer.model --dev_id 0 
```
在读入模型后会显示"Question:"，然后输入就可以了。模型的回答会出现在"Answer"中。结束对话请输入"exit"。

## 3. 支持多会话的Web Demo
我们提供了基于[streamlit](https://streamlit.io/)的web demo，可同时进行多个会话的推理。

### 3.1 参数说明

```bash
usage: web_demo.py [--server.address ADDRESS] [--server.port PORT] 
--server.address: Streamlit 应用服务器的地址；
--server.port: Streamlit 应用服务器的端口号；
```

### 3.2 使用方式
首先安装第三方库
```bash
pip3 install -r python/requirements.txt
```
然后通过streamlit运行web_demo.py即可运行一个web服务

```bash
streamlit run python/web_demo.py --server.address '0.0.0.0' --server.port '9999'
```

命令行输出以下信息则表示启动成功
```bash
  You can now view your Streamlit app in your browser.

  URL: http://0.0.0.0:9999
```

在浏览器中打开输出的地址即可使用，在底部对话框中输入问题。

注意：在docker中启动服务要提前做端口映射，这样才能通过浏览器访问。
