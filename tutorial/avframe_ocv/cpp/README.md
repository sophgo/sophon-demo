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
| 1    | avframe_ocv | avframe到cv::Mat转换的例程 |

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
编译完成后，会在当前目录下生成avframe_ocv.pcie。

### 2.2 SoC平台
通常在x86主机上交叉编译程序，您需要在x86主机上使用SOPHON SDK搭建交叉编译环境，将程序所依赖的头文件和库文件打包至soc-sdk目录中，具体请参考[交叉编译环境搭建](../../../docs/Environment_Install_Guide.md#41-交叉编译环境搭建)。本例程主要依赖libsophon、sophon-opencv和sophon-ffmpeg运行库包。

交叉编译环境搭建好后，使用交叉编译工具链编译生成可执行文件：
```bash
mkdir build && cd build
#请根据实际情况修改-DSDK的路径，需使用绝对路径。
cmake -DTARGET_ARCH=soc -DSDK=/path_to_sdk/soc-sdk ..  
make
```
编译完成后，会在当前目录下生成avframe_ocv.soc。

## 3. 测试
对于PCIe平台，可以直接在PCIe平台上测试；对于SoC平台，需将交叉编译生成的可执行文件及所需的模型、测试数据拷贝到SoC平台中测试。测试的参数及运行方式是一致的，下面主要以PCIe模式进行介绍。

### 3.1 参数说明
可执行程序默认有一套参数，请注意根据实际情况进行传参，`avframe_ocv.pcie与avframe_ocv.soc参数相同。`以avframe_ocv.pcie为例，具体参数说明如下：
```bash
usage:
        ./avframe_ocv.pcie <input_video> [dev_id] 
params:
        input_video: video for decode to get avframe
        dev_id:  using device id, default 0
```
### 3.2 测试视频
测试实例如下
```bash
./avframe_ocv.pcie test_car_person_1080P.mp4 0
```
测试结束后，会将avframe转换后的mat保存到当前文件夹下。

### 3.3 程序说明
本例程通过ffmpeg解码视频得到avframe。

yuv mat是sophon-opencv中mat的扩展数据结构，一般由avframe生成。yuv mat可以通过mat.u->frame得到其中的avframe。

从avframe到普通cv::mat的流程是，从avframe构造yuv mat，yuv mat 通过tomat方法得到普通cv::mat，然后释放yuv mat/avframe。

例程中两个函数是avframe转换到普通cv::mat的不同情况：

avframe_to_cvmat1中，根据avframe生成的yuv mat可以同时管理其中的avframe的释放，当yuv mat调用release的时候，其中的avframe也会自动释放。

avframe_to_cvmat2中，avframe和其生成的yuv mat的释放是相互独立的。此时需要通过调用av_frame_free手动释放avframe，否则会有内存泄漏。
