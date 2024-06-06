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
    * [3.2 测试图片](#32-测试图片)

cpp目录下提供了C++例程以供参考使用，具体情况如下：
| 序号  | C++例程      | 说明                                 |
| ---- | ------------- | -----------------------------------  |
| 1    | ocv_avframe | bgr mat到avframe转换的例程 |

## 1. 环境准备
### 1.1 x86/arm PCIe平台
如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），可以直接使用它作为开发环境和运行环境。您需要安装libsophon、sophon-opencv和sophon-ffmpeg，具体步骤可参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。

### 1.2 SoC平台
如果您使用SoC平台（如SE、SM系列边缘设备），刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包，可直接使用它作为运行环境。通常还需要一台x86主机作为开发环境，用于交叉编译C++程序。


## 2. 程序编译
C++程序运行前需要编译可执行文件。
### 2.1 x86/arm PCIe平台
可以直接在PCIe平台上编译程序：
```bash
mkdir build && cd build
cmake .. 
make
cd ..
```
编译完成后，会在当前目录下生成ocv_avframe.pcie。

### 2.2 SoC平台
通常在x86主机上交叉编译程序，您需要在x86主机上使用SOPHON SDK搭建交叉编译环境，将程序所依赖的头文件和库文件打包至soc-sdk目录中，具体请参考[交叉编译环境搭建](../../../docs/Environment_Install_Guide.md#41-交叉编译环境搭建)。本例程主要依赖libsophon、sophon-opencv和sophon-ffmpeg运行库包。

交叉编译环境搭建好后，使用交叉编译工具链编译生成可执行文件：
```bash
mkdir build && cd build
#请根据实际情况修改-DSDK的路径，需使用绝对路径。
cmake -DTARGET_ARCH=soc -DSDK=/path_to_sdk/soc-sdk ..  
make
```
编译完成后，会在当前目录下生成ocv_avframe.soc。

## 3. 测试
对于PCIe平台，可以直接在PCIe平台上测试；对于SoC平台，需将交叉编译生成的可执行文件及所需的模型、测试数据拷贝到SoC平台中测试。测试的参数及运行方式是一致的，下面主要以PCIe模式进行介绍。

### 3.1 参数说明
可执行程序默认有一套参数，请注意根据实际情况进行传参，`ocv_avframe.pcie与ocv_avframe.soc参数相同。`以ocv_avframe.pcie为例，具体参数说明如下：
```bash
usage:
        ./ocv_avframe.pcie <input_pic> [dev_id]
params:
        input_pic: picture for decode to get bgr mat
        dev_id:  using device id
```
### 3.2 测试视频
测试实例如下
```bash
./ocv_avframe.pcie zidane.jpg 0
```
测试结束后，会将读取的bgr mat格式的图片转换成yuv420p的数据，保存到当前文件夹下，命名为output.yuv。

### 3.3 程序说明
本例程通过opencv解码图片获得bgr mat，然后将bgr mat转为avframe进行编码操作。

yuv mat是sophon-opencv中mat的扩展数据结构，一般由avframe生成。yuv mat可以通过mat.u->frame得到其中的avframe。

bgr mat不能直接转换为avframe，一种比较推荐的做法是先将bgr mat转换为yuv mat，然后取yuv mat中的avframe进行后续操作，此时取得的avframe内容即为对应的bgr mat的内容。

