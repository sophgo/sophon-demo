# Python例程

## 目录 
- [1. 环境准备](#1-环境准备)
  - [1.1 x86/arm PCIe平台](#11-x86arm-pcie平台)
  - [1.2 SoC平台](#12-soc平台)
- [2. 启动服务](#2-启动服务)
  - [2.1 参数说明](#21-参数说明)
  - [2.2 使用方式](#22-使用方式)
  - [2.3 操作说明](#23-使用方式)

python目录下提供了Python例程，具体情况如下：

| 序号  |             Python例程                    |             说明                |
| ---- | ----------------------------------------  | ------------------------------- |
| 1    |          web_demo_st.py                   |         用户交互界面             |

## 1. 环境准备

本例程需要配合sophon-demo/application下的[LLM_api_server](../../LLM_api_server/README.md)项目使用，请确保两个项目置于同一父目录下，并参考[LLM_api_server服务启动](../../LLM_api_server/python/README.md)配置好LLM_api_server所需的环境。以qwen1.5-7b, 2k上下文的模型为例，使用如下指令启动LLM_api_server服务：

```shell
# 进入LLM_api_server可执行脚本例程
cd LLM_api_server/python
# 启动LLM_api_server的OpenAI库调用方式，请确认config.yaml中的模型路径是否正确
python3 api_server.py --config ../../ChatDoc/python/config.yaml
```

### 1.1 x86/arm PCIe平台

如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。

此外您还需要安装其他第三方库：
```bash
pip3 install -r python/requirements.txt
```

您还需要安装sophon-sail，由于本例程需要的sophon-sail版本较新，相关功能还未发布，这里暂时提供一个可用的sophon-sail源码，x86/arm PCIe环境可以通过下面的命令下载：
```bash
pip3 install dfss --upgrade #安装dfss依赖
python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen/sophon-sail.tar.gz
tar xvf sophon-sail.tar.gz
```
参考[sophon-sail编译安装指南](https://doc.sophgo.com/sdk-docs/v24.04.01/docs_latest_release/docs/sophon-sail/docs/zh/html/1_build.html#)编译不包含bmcv,sophon-ffmpeg,sophon-opencv的可被Python3接口调用的Wheel文件。

### 1.2 SoC平台

如果您使用SoC平台（如SE、SM系列边缘设备），并使用它测试本例程，刷机后在`/opt/sophon/`下已经预装了相应的libsophon运行库包。

此外您还需要安装其他第三方库：
```bash
pip3 install -r python/requirements.txt
```
由于本例程需要的sophon-sail版本较新，这里提供一个可用的sophon-sail whl包，SoC环境可以通过下面的命令下载：
```bash
pip3 install dfss --upgrade
python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen/sophon_arm-3.8.0-py3-none-any.whl  #arm soc, py38
```
如果whl包无法使用，也可以参考上一小节，下载源码自己编译。

对于soc系列设备（如SE7/SM7），需要参考如下命令修改设备内存，才能满足所提供的样例模型需要的显存:

```bash
cd /data/
mkdir memedit && cd memedit
wget -nd https://sophon-file.sophon.cn/sophon-prod-s3/drive/23/09/11/13/DeviceMemoryModificationKit.tgz
tar xvf DeviceMemoryModificationKit.tgz
cd DeviceMemoryModificationKit
tar xvf memory_edit_{vx.x}.tar.xz #vx.x是版本号
cd memory_edit
./memory_edit.sh -p #这个命令会打印当前的内存布局信息
./memory_edit.sh -c -npu 7615 -vpu 800 -vpp 800 #npu也可以访问vpu和vpp的内存
sudo cp /data/memedit/DeviceMemoryModificationKit/memory_edit/emmcboot.itb /boot/emmcboot.itb && sync
sudo reboot
```

## 2. 启动服务

python例程不需要编译，可以直接运行，PCIe平台和SoC平台的测试参数和运行方式是相同的。

### 2.1 参数说明

web_demo_st.py使用config.ini配置文件进行参数配置。

config.ini内容如下
```yaml
bce_embedding/bce_reranker: # embedding 和 reranker模型
    bmodel_path: ../models/BM1684X/bce_embedding/bce-embedding-base_v1.bmodel # 模型路径
    token_path: ../models/BM1684X/bce_embedding/token_config  # tokenizer 路径

init_config:
    base_url: http://127.0.0.1:18080/v1/ # LLM_server服务ip和端口
    supported_model: qwen,chatglm3       # 支持的模型

environment_config:
    LLM_MODEL: qwen                      # 默认的LLM模型
    EMBEDDING_MODEL: bce_embedding        # 选择使用的embedding模型
    RERANKER_MODEL: bce_reranker          # 选择使用的reranker模型
    DEVICE_ID: 0                          # 设备号
    server_address: "0.0.0.0"             # streamlit服务ip
    server_port: ""                       # streamlit端口
```

### 2.2 使用方式

```bash
cd python
# 第一次运行时会下载nltk依赖的数据
streamlit run web_demo_st.py
```

### 2.3 操作说明

![UI](<../pics/img1.png>)

### 界面简介
ChatDoc由控制区和聊天对话区组成。控制区用于管理文档和知识库，聊天对话区用于输入和接受消息。

上图中的10号区域是 ChatDoc 当前选中的文档。若10号区域为空，即 ChatDoc 没有选中任何文档，仍在聊天对话区与 ChatDoc 对话，则此时的 ChatDoc 是一个单纯依托 LLM 的 ChatBot。

### 上传文档
点击`1`选择要上传的文档，然后点击按钮`4`构建知识库。随后将embedding文档，完成后将被选中，并显示在10号区域，接着就可开始对话。我们可重复上传文档，embedding成功的文档均会进入10号区域。

### 持久化知识库
10号区域选中的文档在用户刷新或者关闭页面时，将会清空。若用户需要保存这些已经embedding的文档，可以选择持久化知识库，下次进入时无需embedding计算即可加载知识库。具体做法是，在10号区域不为空的情况下，点击按钮`5`即可持久化知识库，知识库的名称是所有文档名称以逗号连接而成。

### 导入知识库

用户可以从选择框`2`查看目前已持久化的知识库。选中我们需要加载的知识库后，点击按钮`3`导入知识库。完成后即可开始对话。注意cpu版的知识库和tpu版的知识库不能混用，若启动tpu版程序，则不能加载已持久化的cpu版知识库；若启动cpu版程序，则不能加载已持久化的tpu版知识库。

### 删除知识库

当用户需要删除本地已经持久化的知识库时，可从选择框`2`选择要删除的知识库，然后点击按钮`6`删除知识库。

### 重命名知识库

![Rename](<../pics/img2.png>)

由于知识库的命名是由其文档的名称组合而来，难免造成知识库名称过长的问题。ChatDoc提供了一个修改知识库名称的功能，选择框`2`选择我们要修改的知识库，然后点击按钮`9`重命名知识库，随后ChatDoc将弹出一个输入框和一个确认按钮，如上图。在输出框输入修改后的名称，然后点击`确认重命名`按钮。

### 清除聊天记录

点击按钮`7`即可清除聊天对话区聊天记录。其他不受影响。

### 移除选中文档

点击按钮`8`将清空10号区域，同时清除聊天记录。