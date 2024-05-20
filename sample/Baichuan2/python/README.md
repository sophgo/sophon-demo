# Python例程

## 目录

* [1. 环境准备](#1-环境准备)
    * [1.1 x86/arm PCIe平台](#11-x86arm-pcie平台)
    * [1.2 SoC平台](#12-soc平台)
* [2. 推理测试](#2-推理测试)
    * [2.1 参数说明](#21-参数说明)
    * [2.2 使用方式](#22-使用方式)
* [3. 支持多会话的Web Demo](#3-支持多会话的Web-Demo)
    * [3.1 使用方式](#31-使用方式)
    * [3.2 程序流程图](#32-程序流程图)
    * [3.3 程序二次开发说明](#33-程序二次开发说明)

python目录下提供了一系列Python例程，具体情况如下：

| 序号 |  Python例程       | 说明                                |
| ---- | ---------------- | -----------------------------------  |
| 1    | baichuan2.py     | 使用SAIL推理 |
| 2    | web_demo.py     | 支持多会话的web demo |


## 1. 环境准备
### 1.1 x86/arm PCIe平台
如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon、sophon-opencv、sophon-ffmpeg，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。

其中，libsophon需要使用0.5.1版本，相关功能暂未发布，这里暂时提供一个可用的libsophon版本，您可以通过下面的命令下载：
```bash
pip3 install dfss --upgrade  #安装dfss依赖
python3 -m dfss --url=open@sophgo.com:sophon-demo/baichuan2/libsophon/libsophon.tar.gz  #下载libsophon软件包
tar xvf libsophon.tar.gz
```
下载完libsophon后，可参考[libsophon用户手册](https://doc.sophgo.com/sdk-docs/v23.09.01-lts/docs_latest_release/docs/libsophon/guide/html/1_install.html#)，根据对应的Linux发行版进行安装。

此外您还需要安装其他第三方库：
```bash
pip3 install -r python/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
您还需要安装sophon-sail，由于本例程需要的sophon-sail版本为3.8.0，相关功能暂未发布，这里暂时提供一个可用的sophon-sail版本，x86 PCIe环境可以通过下面的命令下载：
```bash
#x86 pcie, py38
python3 -m dfss --url=open@sophgo.com:sophon-demo/baichuan2/sail/sophon-3.8.0-py3-none-any.whl
pip3 install sophon-3.8.0-py3-none-any.whl
```
如果您需要其他版本的sophon-sail，或者遇到glibc版本问题（pcie环境常见），可以通过以下命令下载源码，并参考[sophon-sail编译安装指南](https://doc.sophgo.com/sdk-docs/v23.07.01/docs_latest_release/docs/sophon-sail/docs/zh/html/1_build.html#)自己编译sophon-sail。
```bash
python3 -m dfss --url=open@sophgo.com:sophon-demo/baichuan2/sail/sophon-sail_3.8.0.tar.gz
tar xvf sophon-sail_3.8.0.tar.gz
```

### 1.2 SoC平台

如果您使用SoC平台（如SE、SM系列边缘设备），并使用它测试本例程，刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包。

此外您还需要安装其他第三方库：
```bash
pip3 install -r python/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
由于本例程需要的sophon-sail版本为3.8.0，这里提供一个可用的sophon-sail whl包，SoC环境可以通过下面的命令下载：
```bash
pip3 install dfss --upgrade
python3 -m dfss --url=open@sophgo.com:sophon-demo/baichuan2/sail/sophon_arm-3.8.0-py3-none-any.whl #arm soc, py38
```
如果您需要其他版本的sophon-sail，或者遇到glibc版本问题，可以通过以下命令下载libsophon软件包和sophon-sail源码，并参考[libsophon安装指南](https://doc.sophgo.com/sdk-docs/v23.09.01-lts/docs_latest_release/docs/libsophon/guide/html/1_install.html#)自己安装libsophon，然后参考[sophon-sail编译安装指南](https://doc.sophgo.com/sdk-docs/v23.07.01/docs_latest_release/docs/sophon-sail/docs/zh/html/1_build.html#)自己编译sophon-sail。
```bash
#libsophon软件包
python3 -m dfss --url=open@sophgo.com:sophon-demo/baichuan2/libsophon/libsophon_soc_0.5.1_aarch64.tar.gz
tar xvf libsophon_soc_0.5.1_aarch64.tar.gz

#sophon-sail源码
python3 -m dfss --url=open@sophgo.com:sophon-demo/baichuan2/sail/sophon-sail_3.8.0.tar.gz
tar xvf sophon-sail_3.8.0.tar.gz
```

## 2. 推理测试
python例程不需要编译，可以直接运行，PCIe平台和SoC平台的测试参数和运行方式是相同的。
### 2.1 参数说明

```bash
usage: baichuan2.py [--bmodel BMODEL] [--token TOKEN] [--dev_ids DEV_ID]
--bmodel: 用于推理的bmodel路径；
--token: tokenizer目录路径；
--dev_ids: 用于推理的tpu设备id，多个设备id用空格隔开；
--help: 输出帮助信息
```

### 2.2 使用方式

```bash
python3 python/baichuan2.py --bmodel models/BM1684X/baichuan2-7b_int8_1dev.bmodel --token python/token_config --dev_ids 0
```
在读入模型后会显示"Question:"，然后输入就可以了。模型的回答会出现在"Answer"中。结束对话请输入"exit"。

## 3. 支持多会话的Web Demo
我们提供了基于[streamlit](https://streamlit.io/)的web demo，可同时进行多个会话的推理。

### 3.1 使用方式
首先安装第三方库
```bash
pip3 install -r python/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
然后通过streamlit运行web_demo.py即可运行一个web_server

```bash
streamlit run python/web_demo.py
```

首次运行需要输入邮箱，输入邮箱后命令行输出以下信息则表示启动成功
```bash
 You can now view your Streamlit app in your browser.

  Network URL: http://172.xx.xx.xx:8501
  External URL: http://103.xx.xxx.xxx:8501
```

在浏览器中打开输出的地址即可使用，web页面如下，在底部对话框中输入问题。
![diagram](../pics/web_demo.png)

### 3.2 程序流程图
通过将同一个sail.EngineLLM, tokenizer传入不同Baichuan2实例对象中，从而实现多会话同时推理的能力，具体流程如下：
![diagram](../pics/multi_session.jpg)

其中chat_stream接收用户输入的问题input和历史消息history，通过sail.EngineLLM进行推理，流式返回生成的文本，具体过程如下：
![diagram](../pics/chat_stream.png)

### 3.3 程序二次开发说明

查看web_demo.py的7-9行，参数说明如下：
```python
token_path = './python/token_config'
bmodel_path = './models/BM1684X/baichuan2-7b_int8_1dev.bmodel'
dev_id = [0]
```
```bash
bmodel_path: 用于推理的bmodel路径；
token_path: tokenizer目录路径；
dev_id: 用于推理的tpu设备id列表；
```
通过修改对应参数可以改变demo的bmodel，tokenizer，dev_id。

当用户输入问题并提交后，程序会创建一个Baichuan2实例，并开始推理过程，代码在web_demo.py中的52和57行
```python
client = Baichuan2(st.session_state.handle, st.session_state.engine, st.session_state.tokenizer)
```
```python
stream = client.chat_stream(input = prompt,history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages])
```

Baichuan2实例的创建需要接收sail.Handle，sail.EngineLLM，tokenizer，通过上面的bmodel_path， token_path， dev_id三个参数来控制。并实现了一个推理接口`chat_stream(input, history)`  input是用户输入的问题，history是历史消息。例如：
```python
input='用c++实现一个冒泡排序'
history = [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么我可以帮助你的吗？"},
]
```

如果其他的模型的创建也可以接收sail.Handle，sail.EngineLLM，tokenizer参数，并且实现了类似`chat_stream(input, history)` 的流式推理接口，则可以替换相应的模型。例如用Qwen进行替换：
```python
client = Qwen(st.session_state.handle, st.session_state.engine, st.session_state.tokenizer)
...
stream = client.chat_stream(...)
```