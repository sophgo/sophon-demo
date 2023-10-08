# C++例程

## 目录

- [C++例程](#c例程)
  - [目录](#目录)
  - [1. x86/arm PCIe平台 编译、运行](#1-x86arm-pcie平台-编译运行)
    - [1.1 环境配置](#11-环境配置)
    - [1.2 程序编译](#12-程序编译)
    - [1.3 参数说明](#13-参数说明)
    - [1.4 运行测试](#14-运行测试)
  - [2 SoC平台 编译、运行](#2-soc平台-编译运行)
    - [2.1 环境配置](#21-环境配置)
    - [2.2 程序编译](#22-程序编译)
    - [2.3 程序参数说明](#23-程序参数说明)
    - [3.2 运行测试](#32-运行测试)


## 1. x86/arm PCIe平台 编译、运行
### 1.1 环境配置
如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），可以直接使用它作为开发环境和运行环境。您需要安装libsophon，具体步骤可参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。

### 1.2 程序编译
C++程序运行前需要编译可执行文件，可以直接在PCIe平台上编译程序：

```bash
mkdir build && cd build
cmake .. && make 
cd ..
```
编译完成后，会在cpp目录下生成chatglm2.pcie。

### 1.3 参数说明
可执行程序默认有一套参数，请注意根据实际情况进行传参，具体参数说明如下：

```bash
usage:./chatglm2.pcie [params]
        --bmodel (value:../models/BM1684X/chatglm2-6b.bmodel)
                bmodel file path
        --token (value:../models/BM1684X/tokenizer.model)
                token file path
        --dev_id (value:0)
                TPU device id
        --help (value:0)
                print help information.
```
**注意：** CPP传参与python不同，需要用等于号，例如`./chatglm2.pcie --bmodel=xxx`。

### 1.4 运行测试
```bash
./chatglm2.pcie --bmodel=../models/BM1684X/chatglm2-6b.bmodel --token=../models/BM1684X/tokenizer.model --dev_id=0
```


## 2 SoC平台 编译、运行
### 2.1 环境配置
如果您使用SoC平台（目前可支持SE7），刷机后在`/opt/sophon/`下已经预装了相应的libsophon运行库包，可直接使用它作为运行环境。通常还需要一台x86主机作为开发环境，用于交叉编译C++程序。

### 2.2 程序编译
通常在x86主机上交叉编译程序，您需要在x86主机上使用SOPHON SDK搭建交叉编译环境，将程序所依赖的头文件和库文件打包至soc-sdk目录中，具体请参考[交叉编译环境搭建](../../../docs/Environment_Install_Guide.md#41-交叉编译环境搭建)。本例程主要依赖libsophon运行库包。

交叉编译环境搭建好后，使用交叉编译工具链编译生成可执行文件：

```bash
mkdir build && cd build
#请根据实际情况修改-DSDK的路径，需使用绝对路径。
cmake -DTARGET_ARCH=soc -DSDK=/path/sdk-soc ..  
make
cd ..
```
编译完成后，会在cpp目录下生成chatglm2.soc。

### 2.3 程序参数说明
对于SoC平台，需将交叉编译生成的可执行文件及所需的模型、测试数据拷贝到SoC平台中测试。测试参数与PCIE是一致的。


### 3.2 运行测试
```bash
./chatglm2.soc --bmodel=../models/BM1684X/chatglm2-6b.bmodel --token=../models/BM1684X/tokenizer.model --dev_id=0
```