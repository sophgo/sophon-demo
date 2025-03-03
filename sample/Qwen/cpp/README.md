[简体中文](./README.md)

# C++例程

## 目录

- [C++例程](#c例程)
  - [目录](#目录)
  - [1. 环境准备](#1-环境准备)
    - [1.1 x86/arm/riscv PCIe平台](#11-x86armriscv-pcie平台)
    - [1.2 SoC平台](#12-soc平台)
  - [2. 程序编译](#2-程序编译)
    - [2.1 x86/arm/riscv PCIe平台](#21-x86armriscv-pcie平台)
    - [2.2 SoC平台](#22-soc平台)
  - [3. 推理测试](#3-推理测试)
    - [3.1 参数说明](#31-参数说明)
    - [3.2 测试](#32-测试)

cpp目录下提供了C++例程以供参考使用，具体情况如下：
| 序号  | C++例程      | 说明                                 |
| ---- | ------------- | -----------------------------------  |
| 1    | qwen_bmlib     |         使用BMRT推理                 |

## 1. 环境准备
### 1.1 x86/arm/riscv PCIe平台
如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），可以直接使用它作为开发环境和运行环境。您需要安装libsophon，具体步骤可参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)或[riscv-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#6-riscv-pcie平台的开发和运行环境搭建)。

### 1.2 SoC平台
如果您使用SoC平台（如SE、SM系列边缘设备），刷机后在`/opt/sophon/`下已经预装了相应的libsophon运行库包，可直接使用它作为运行环境。通常还需要一台x86主机作为开发环境，用于交叉编译C++程序。


## 2. 程序编译
C++程序运行前需要编译可执行文件。
### 2.1 x86/arm/riscv PCIe平台
可以直接在PCIe平台上编译程序：
#### 2.1.1 BMLIB接口程序
如果您使用BMRT裸接口，需要执行如下步骤。
```bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen/cpp/third_party.zip
unzip third_party.zip
rm -f third_party.zip

cd qwen_bmlib
mkdir build && cd build
cmake ..
make
cd ..
```
编译完成后，会在qwen_bmlib目录下生成qwen_bmlib.pcie。

### 2.2 SoC平台
通常在x86主机上交叉编译程序，您需要在x86主机上使用SOPHON SDK搭建交叉编译环境，将程序所依赖的头文件和库文件打包至soc-sdk目录中，具体请参考[交叉编译环境搭建](../../../docs/Environment_Install_Guide.md#41-交叉编译环境搭建)。本例程主要依赖libsophon运行库包。

交叉编译环境搭建好后，使用交叉编译工具链编译生成可执行文件。

#### 2.2.1 编译Tokenizer
若您想使用预编译好的Tokenizer可跳过该小节。

- 下载源码
```bash
git clone https://github.com/mlc-ai/tokenizers-cpp
cd tokenizers-cpp
```

- 升级RUST编译工具包
```bash
export RUSTUP_DIST_SERVER="https://rsproxy.cn"
export RUSTUP_UPDATE_ROOT="https://rsproxy.cn/rustup"
curl --proto '=https' --tlsv1.2 -sSf https://rsproxy.cn/rustup-init.sh | sh
```

- 直接在目标平台编译
```bash
mkdir build && cd build
cmake ..
make
```

- 拷贝静态库和头文件`build/libtokenizers_c.a`、`build/libtokenizers_cpp.a`、`build/sentencepiece/libsentencepiece.a`、`build/release/libtokenizers_c.a`和`/include`到工作目录，即可

#### 2.2.2 BMLIB接口程序
如果您使用BMRT裸接口，需要执行如下步骤。
```bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen/cpp/third_party.zip
unzip third_party.zip
rm -f third_party.zip

cd qwen_bmlib
mkdir build && cd build
#请根据实际情况修改-DSDK和-DSAIL_PATH的路径，需使用绝对路径。
cmake -DTARGET_ARCH=soc -DSDK=/path_to_sdk/soc-sdk ..
make
cd ..
```
编译完成后，会在qwen_bmlib目录下生成qwen_bmlib.soc。

## 3. 推理测试
对于PCIe平台，可以直接在PCIe平台上推理测试；对于SoC平台，需将交叉编译生成的可执行文件及所需的模型、测试数据拷贝到SoC平台中测试。测试的参数及运行方式是一致的，下面主要以PCIe模式进行介绍。

### 3.1 参数说明
可执行程序默认有一套参数，请注意根据实际情况进行传参，以qwen_bmlib.pcie为例，具体参数说明如下：
```bash
Usage: qwen_bmlib.pcie [params] 

  --help                  : Show help info.
  --bmodel_path           : Set bmodel path 
  --tokenizer_path        : Set tokenizer path 
  --dev_id                : Set devices to run for model, e.g. 1,2, if not provided, use 0
```
> **注意：** cpp例程传参与python不同，需要用等于号，例如`./qwen_bmlib.pcie --bmodel_path=xxx`。

### 3.2 测试
需要在`cpp/qwen_bmlib`目录下执行程序，视频测试实例如下。
```bash
./qwen_bmlib.pcie
```