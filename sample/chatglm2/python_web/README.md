[简体中文](./README.md)

# Python_web例程

## 目录

- [Python\_web例程](#python_web例程)
  - [目录](#目录)
  - [1. 环境准备](#1-环境准备)
    - [1.1 x86/arm PCIe平台](#11-x86arm-pcie平台)
    - [1.2 SoC平台](#12-soc平台)
  - [2. 程序编译](#2-程序编译)
    - [2.1 x86/arm PCIe平台](#21-x86arm-pcie平台)
    - [2.2 SoC平台](#22-soc平台)
  - [3. 推理测试](#3-推理测试)
    - [3.1 参数说明](#31-参数说明)
    - [3.2 运行测试](#32-运行测试)


## 1. 环境准备
### 1.1 x86/arm PCIe平台

如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。

web版本需要安装一些依赖：
```
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/ gradio --upgrade
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/ numpy==1.20.3
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/ mdtex2html
```

### 1.2 SoC平台

如果您使用SoC平台（如SE、SM系列边缘设备），并使用它测试本例程，刷机后在`/opt/sophon/`下已经预装了相应的libsophon运行库包。

web版本需要安装一些依赖：
```
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/ gradio --upgrade
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/ numpy==1.20.3
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/ mdtex2html
```

## 2. 程序编译
C++程序运行前需要编译可执行文件。
### 2.1 x86/arm PCIe平台
可以直接在PCIe平台上编译程序：

```bash
mkdir build && cd build
cmake .. && make # 生成chatglm2.pcie
cd ../..
```
编译完成后，会在python目录下生成ChatGLM2.cpython-38-x86_64-linux-gnu.so。

### 2.2 SoC平台
通常在x86主机上交叉编译程序，您需要在x86主机上使用SOPHON SDK搭建交叉编译环境，将程序所依赖的头文件和库文件打包至soc-sdk目录中，具体请参考[交叉编译环境搭建](../../../docs/Environment_Install_Guide.md#41-交叉编译环境搭建)。本例程主要依赖libsophon运行库包。

交叉编译环境搭建好后，使用交叉编译工具链编译生成可执行文件：

```bash
mkdir build && cd build
#请根据实际情况修改-DSDK的路径，需使用绝对路径。
cmake -DTARGET_ARCH=soc -DSDK=/path/sdk-soc ..  
make
cd ../..
```
编译完成后，会在python目录下生成ChatGLM2.cpython-38-x86_64-linux-gnu.so。

## 3. 推理测试
对于PCIe平台，可以直接在PCIe平台上推理测试；对于SoC平台，需将交叉编译生成的可执行文件及所需的模型、测试数据拷贝到SoC平台中测试。测试的参数及运行方式是一致的，下面主要以PCIe模式进行介绍。

### 3.1 参数说明
可执行程序默认有一套参数，请注意根据实际情况进行传参，具体参数说明如下：

```bash
usage: chatglm2.py [--bmodel BMODEL] [--token TOKEN] [--dev_id DEV_ID]
--bmodel: 用于推理的bmodel路径。
--token: 用于推理的token路径。
--dev_id: 用于推理的tpu设备id；
```

### 3.2 运行测试

```bash 
python3 python_web/web_chatglm2.py --bmodel models/BM1684X/chatglm2-6b.bmodel --token models/BM1684X/tokenizer.model --dev_id 0
```