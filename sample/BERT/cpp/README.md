# C++例程

## 目录

* [1. 环境准备](#1-环境准备)
    * [1.1 x86/arm PCIe平台](#11-x86arm-pcie平台)
    * [1.2 SoC平台](#12-soc平台)
* [2. 程序编译](#2-程序编译)
    * [2.1 x86/arm PCIe平台](#21-x86arm-pcie平台)
    * [2.2 SoC平台](#22-soc平台)
* [3. 推理测试](#3-推理测试)
    * [3.1 参数说明](#31-参数说明)
    * [3.2 测试文本](#32-测试文本)
    * [3.3 测试数据集](#33-测试数据集)

cpp目录下提供了C++例程以供参考使用，具体情况如下：
| 序号  | C++例程      | 说明                                 |
| ---- | ------------- | -----------------------------------  |
| 1    | bert_sail   | 使用SAIL推理                           |

## 1. 环境准备
### 1.1 x86/arm PCIe平台
如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），可以直接使用它作为开发环境和运行环境。您需要安装libsophon、sophon-opencv和sophon-ffmpeg，具体步骤可参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。
本例程依赖boost>=1.65,
```
apt install libboostx.xx-dev
```
x.xx为对应版本。可通过
```
sudo apt-cache search libboost
```
查看
以ubuntu20.04为例
```
apt install libboost1.71-dev
```
### 1.2 SoC平台
如果您使用SoC平台（如SE、SM系列边缘设备），刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包，可直接使用它作为运行环境。通常还需要一台x86主机作为开发环境，用于交叉编译C++程序。
本例程依赖boost>=1.65,
```
apt install libboostx.xx-dev
```
x.xx为对应版本。可通过
```
sudo apt-cache search libboost
```
查看
以ubuntu20.04为例
```
apt install libboost1.71-dev
```
## 2. 程序编译
C++程序运行前需要编译可执行文件。
### 2.1 x86/arm PCIe平台
可以直接在PCIe平台上编译程序：
#### 2.1.1 sail
```bash
cd cpp/bert_sail
mkdir build && cd build
cmake .. 
make
cd ..
```
编译完成后，会在bert_sail目录下生成bert_sail.pcie。


### 2.2 SoC平台
通常在x86主机上交叉编译程序，您需要在x86主机上使用SOPHON SDK搭建交叉编译环境，将程序所依赖的头文件和库文件打包至soc-sdk目录中，具体请参考[交叉编译环境搭建](../../../docs/Environment_Install_Guide.md#41-交叉编译环境搭建)。本例程主要依赖libsophon、sophon-opencv和sophon-ffmpeg运行库包。

交叉编译环境搭建好后，使用交叉编译工具链编译生成可执行文件：

#### 2.2.1 sail
如果您使用sophon-sail接口，需要参考[交叉编译安装sophon-sail](../../../docs/Environment_Install_Guide.md#42-交叉编译安装sophon-sail)，给soc环境配置sophon-sail，然后进行如下步骤。
```bash
cd cpp/bert_sail
mkdir build && cd build
#请根据实际情况修改-DSDK和-DSAIL_PATH的路径，需使用绝对路径。
cmake -DTARGET_ARCH=soc -DSDK=/path_to_sdk/soc-sdk -DSAIL_PATH=/path_to_sail/sophon-sail/build_soc/sophon-sail ..
make
```
编译完成后，会在bert_sail目录下生成bert_sail.soc。

## 3. 推理测试
对于PCIe平台，可以直接在PCIe平台上推理测试；对于SoC平台，需将交叉编译生成的可执行文件及所需的模型、测试数据拷贝到SoC平台中测试。测试的参数及运行方式是一致的，下面主要以PCIe模式进行介绍。

### 3.1 参数说明
可执行程序默认有一套参数，请注意根据实际情况进行传参，以bert_sail.pcie为例，具体参数说明如下：
```bash
Usage: bert_sail.pcie [params]

        --bmodel (value: ../../models/BM1684/bert4torch_output_fp32_1b.bmodel)
                bmodel file path
        --dev_id (value:0)
                TPU device id
        --help (value:true)
                print help information.
        --input (value:"../../datasets/china-people-daily-ner-corpus/example.test")
                input dev or text_path or dateset path        
        --dict_path (value: ../../data/pre_train/chinese-bert-wwm/vocab.txt)
                pre_train vocab file
      
```
**注意：** CPP传参与python不同，需要用等于号，例如`./bert_sail.pcie --bmodel=xxx`。

### 3.2 测试文本
文本测试实例如下，支持对文件进行测试。
```bash
./bert_sail.pcie --input=../../datasets/china-people-daily-ner-corpus/test.txt --bmodel=../../models/BM1684/bert4torch_output_fp32_1b.bmodel --dev_id=0 
```
测试结束后，会打印预测结果等信息。

### 3.3 测试数据集
文本测试实例如下，支持对文件进行测试。
```bash
./bert_sail.pcie --input=../../datasets/china-people-daily-ner-corpus/example.test --bmodel=../../models/BM1684/bert4torch_output_fp32_1b.bmodel --dev_id=0 
```
测试结束后，预测的结果保存在`results/bert4torch_output_fp32_1b.bmodel_test_sail_cpp_result.txt`下，同时会打印预测结果、推理时间等信息。

>**注意：**

若在SoC模式下执行报错:
```bash
./bert_sail.soc: error while loading shared libraries: libsail.so: cannot open shared object file: No such file or directory
```
请设置如下环境变量:
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/sophon/sophon-sail/lib
```



