# C++例程
* [1. 环境准备](#1-环境准备)
    * [1.1 x86/arm PCIe平台](#11-x86arm-pcie平台)
    * [1.2 SoC平台](#12-soc平台)
* [2. 程序编译](#2-程序编译)
    * [2.1 x86/arm PCIe平台](#21-x86arm-pcie平台)
    * [2.2 SoC平台](#22-soc平台)
* [3. 推理测试](#3-推理测试)
    * [3.1 参数说明](#31-参数说明)
    * [3.2 测试图片](#32-测试图片)


cpp目录下提供了一系列C++例程以供参考使用，具体情况如下：
| 序号  | C++例程      | 说明                                 |
| ---- | ------------- | -----------------------------------  |
| 1    | lprnet_opencv | 使用OpenCV解码、OpenCV前处理、BMRT推理 |
| 2    | lprnet_bmcv   | 使用FFmpeg解码、BMCV前处理、BMRT推理   |


## 1. 环境准备
### 1.1 x86/arm PCIe平台
如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），可以直接使用它作为开发环境和运行环境。您需要安装libsophon、sophon-opencv和sophon-ffmpeg，具体步骤可参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。

### 1.2 SoC平台
如果您使用SoC平台（如SE、SM系列边缘设备），刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包，可直接使用它作为运行环境。通常还需要一台x86主机作为开发环境，用于交叉编译C++程序。


## 2. 程序编译
C++程序运行前需要编译可执行文件。
### 2.1 x86/arm PCIe平台
可以直接在PCIe平台上编译程序,lprnet_opencv和lprnet_bmcv编译方法相同，以编译lprnet_opencv程序为例：
```bash
cd cpp/lprnet_opencv
mkdir build && cd build
cmake ..
make
cd ..
```
编译完成后，会在lprnet_opencv目录下生成lprnet_opencv.pcie。

### 2.2 SoC平台
通常在x86主机上交叉编译程序，您需要在x86主机上使用SOPHON SDK搭建交叉编译环境，将程序所依赖的头文件和库文件打包至soc-sdk目录中，具体请参考[交叉编译环境搭建](../../../docs/Environment_Install_Guide.md#41-交叉编译环境搭建)。本例程主要依赖libsophon、sophon-opencv和sophon-ffmpeg运行库包。

交叉编译环境搭建好后，使用交叉编译工具链编译生成可执行文件，lprnet_opencv和lprnet_bmcv编译方法相同，以编译lprnet_opencv程序为例：
```bash
cd cpp/lprnet_opencv
mkdir build && cd build
#请根据实际情况修改-DSDK的路径，需使用绝对路径。
cmake -DTARGET_ARCH=soc -DSDK=/path_to_sdk/soc-sdk ..  
make
```
编译完成后，会在lprnet_opencv目录下生成lprnet_opencv.soc。

## 3. 推理测试
对于PCIe平台，可以直接在PCIe平台上推理测试；对于SoC平台，需将交叉编译生成的可执行文件及所需的模型、测试数据拷贝到SoC平台中测试。测试的参数及运行方式是一致的，下面主要以PCIe模式进行介绍。

### 3.1 参数说明
可执行程序默认有一套参数，请注意根据实际情况进行传参，lprnet_opencv和lprnet_bmcv参数相同，以lprnet_opencv为例，具体参数说明如下：

```bash
Usage: lprnet_opencv.pcie [params]

        --bmodel (value:../../models/BM1684/lprnet_fp32_1b.bmodel)
                bmodel file path
        --dev_id (value:0)
                TPU device id
        --help (value:true)
                print help information.
        --input (value:../../datasets/test)
                input path, images direction or video file path
```
**注意：** CPP传参与python不同，需要用等于号，例如`./lprnet_opencv.pcie --bmodel=xxx`。

### 3.2 测试图片
图片测试实例如下，支持对整个图片文件夹进行测试。
```bash
./lprnet_opencv.pcie --input=../../datasets/test --bmodel=../../models/BM1684X/lprnet_fp32_1b.bmodel --dev_id=0
```

执行完成后，会将预测结果保存在`results/lprnet_fp32_1b.bmodel_test_opencv_cpp_result.json`下，同时会打印预测结果、推理时间等信息，输出如下：

```bash
......
豫RM6396.jpg pred:皖RM6396
闽D33U29.jpg pred:皖D33U29
鲁AW9V20.jpg pred:鲁AW9V20
鲁BE31L9.jpg pred:鲁BE31L9
鲁Q08F99.jpg pred:鲁Q08F99
鲁R8D57Z.jpg pred:鲁R8D57Z
================
result saved in results/lprnet_fp32_1b.bmodel_test_opencv_cpp_result.json
================
infer_time = 0.745000ms
QPS = 625

############################
SUMMARY: lprnet detect
############################
[      lprnet overall]  loops:    1 avg: 1600402 us
[          read image]  loops:  100 avg: 388 us
[           detection]  loops:  100 avg: 1178 us
[  lprnet pre-process]  loops:  100 avg: 151 us
[    lprnet inference]  loops:  100 avg: 745 us
[ lprnet post-process]  loops:  100 avg: 238 us

```

