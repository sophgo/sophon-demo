[简体中文](./README.md)

# Python例程

## 目录

- [Python例程](#python例程)
  - [目录](#目录)
  - [1. x86/arm PCIe平台 编译、运行](#1-x86arm-pcie平台-编译运行)
    - [1.1 环境配置](#11-环境配置)
    - [1.2 程序编译](#12-程序编译)
    - [1.3 程序参数说明](#13-程序参数说明)
    - [1.4 运行测试](#14-运行测试)
  - [2 SoC平台 交叉编译、运行](#2-soc平台-交叉编译运行)
    - [2.1 环境配置](#21-环境配置)
    - [2.2 程序编译](#22-程序编译)
    - [2.3 程序参数说明](#23-程序参数说明)
    - [2.4 运行测试](#24-运行测试)
  - [3 SoC平台 编译、运行](#3-soc平台-编译运行)
    - [3.1 环境配置](#31-环境配置)
    - [3.2 程序编译](#32-程序编译)
    - [3.3 程序参数说明](#33-程序参数说明)
    - [3.4 运行测试](#34-运行测试)


## 1. x86/arm PCIe平台 编译、运行
### 1.1 环境配置

如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。

python版本需要pybind11，需要安装pybind11，并检查一下CMakeLists.txt的pybind11路径是否正确，
```
# 安装pybind11
sudo pip3 install pybind11
# 查找当前pybind11路径，并添加到CMakeLists.txt
sudo find / -name pybind11
```
### 1.2 程序编译
C++程序运行前需要编译可执行文件，可以直接在PCIe平台上编译程序：

```bash
mkdir build && cd build
cmake .. && make # 生成chatglm2.pcie
cd ..
```
编译完成后，会在python目录下生成ChatGLM2.cpython-38-x86_64-linux-gnu.so。

### 1.3 程序参数说明
对于PCIe平台，可以直接在PCIe平台上推理测试；可执行程序默认有一套参数，请注意根据实际情况进行传参，具体参数说明如下：

```bash
usage: chatglm2.py [--bmodel BMODEL] [--token TOKEN] [--dev_id DEV_ID]
--bmodel: 用于推理的bmodel路径;
--token: 用于推理的token路径;
--dev_id: 用于推理的tpu设备id。
```

### 1.4 运行测试

```bash 
python3 chatglm2.py --bmodel ../models/BM1684X/chatglm2-6b_f16.bmodel --token ../models/BM1684X/tokenizer.model --dev_id 0
```


## 2 SoC平台 交叉编译、运行

如果您使用SoC平台（目前可支持SE7），刷机后在`/opt/sophon/`下已经预装了相应的libsophon运行库包，可直接使用它作为运行环境。通常还需要一台x86主机作为开发环境，用于交叉编译C++程序。
### 2.1 环境配置
python版本需要pybind11，需要安装pybind11，并检查一下CMakeLists.txt的pybind11路径是否正确，
```
# 安装pybind11
sudo pip3 install pybind11
# 查找当前pybind11路径，并添加到CMakeLists.txt
sudo find / -name pybind11
```

### 2.2 程序编译
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

### 2.3 程序参数说明
对于SoC平台，需将交叉编译生成的可执行文件及所需的模型、测试数据拷贝到SoC平台中测试。测试参数与PCIE是一致的。

### 2.4 运行测试

```bash 
python3 chatglm2.py --bmodel ../models/BM1684X/chatglm2-6b_f16.bmodel --token ../models/BM1684X/tokenizer.model --dev_id 0
```


## 3 SoC平台 编译、运行

### 3.1 环境配置
如果您使用SoC平台（目前可支持SE7），刷机后在`/opt/sophon/`下已经预装了相应的libsophon运行库包，可直接使用它作为运行环境。如果将其作为开发环境，需要进入`/home/linaro/bsp-debs`路径下，安装开发包。
```
sudo dpkg -i sophon-soc-libsophon-dev_x.y.z_arm64.deb
```

python版本需要pybind11，需要安装pybind11，并检查一下CMakeLists.txt的pybind11路径是否正确，
```
# 安装pybind11
sudo pip3 install pybind11
# 查找当前pybind11路径，并添加到CMakeLists.txt
sudo find / -name pybind11
```

### 3.2 程序编译
本例程主要依赖libsophon运行库包,在SOC平台上安装好开发环境后，使用下面的命令编译生成可执行文件：

```bash
mkdir build && cd build
cmake -DTARGET_ARCH=soc_base ..  
make
cd ..
```
编译完成后，会在python目录下生成ChatGLM2.cpython-38-x86_64-linux-gnu.so。

### 3.3 程序参数说明
对于SoC平台，测试参数与PCIE是一致的。

### 3.4 运行测试

```bash 
python3 chatglm2.py --bmodel ../models/BM1684X/chatglm2-6b_f16.bmodel --token ../models/BM1684X/tokenizer.model --dev_id 0
```