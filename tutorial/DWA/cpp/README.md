# DWA C++例程

## 目录

- [DWA C++例程](#dwa-c例程)
  - [目录](#目录)
  - [1. 环境准备](#1-环境准备)
  - [2. 程序编译](#2-程序编译)
  - [3. 测试](#3-测试)
    - [3.1 参数说明](#31-参数说明)
    - [3.2 测试图片](#32-测试图片)

提供了C++例程以供参考使用，具体情况如下：
| 序号  | C++例程      | 说明                                 |
| ---- | ------------- | -----------------------------------  |
| 1    | [dwa_bmcv](./dwa_bmcv) | 使用BMCV接口做图像矫正 |

## 1. 环境准备

SoC平台（如SE、SM系列边缘设备），刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包，可直接使用它作为运行环境。通常还需要一台x86主机作为开发环境，用于交叉编译C++程序。

## 2. 程序编译
C++程序运行前需要编译可执行文件，目前该功能只支持在BM1688/CV186X上使用。

通常在x86主机上交叉编译程序，您需要在x86主机上使用SOPHON SDK搭建交叉编译环境，将程序所依赖的头文件和库文件打包至soc-sdk目录中，具体请参考[交叉编译环境搭建](../../../docs/Environment_Install_Guide.md#41-交叉编译环境搭建)。本例程主要依赖libsophon、sophon-opencv和sophon-ffmpeg运行库包。

交叉编译环境搭建好后，使用交叉编译工具链编译生成可执行文件：

```bash
cd cpp/dwa_bmcv
mkdir build && cd build
#请根据实际情况修改-DSDK的路径，需使用绝对路径。
cmake -DTARGET_ARCH=soc -DSDK=/path_to_sdk/soc-sdk ..  
make
```
编译完成后，会在dwa_bmcv目录下生成dwa_bmcv.soc.


## 3. 测试
对于SoC平台，需将交叉编译生成的可执行文件及所需的模型、测试数据拷贝到SoC平台中测试。测试的参数及运行方式是一致的，下面主要以PCIe模式进行介绍。

### 3.1 参数说明
可执行程序默认有一套参数，请注意根据实际情况进行传参。
```bash
Usage: 
./dwa_bmcv.soc <input_grid_path> <input_image_path> <if_resize> <resize_h> <resize_w> <debug>
``` 
注意：
1. <resize>参数是指在dwa之前对图片缩放，是否resize需要根据相机标定时是否缩放确定，即配合参数<input_grid_path>确认。
2. <debug>参数是指是否保存dwa的输入输出图像。


### 3.2 测试图片
图片测试实例如下
```bash
./dwa_bmcv.soc 
```

```bash
./dwa_bmcv.soc 
    --input_grid_path=/data/images/left/LL.dat
    --input_image_path=/data/images/left/left.jpg
    --if_resize=true
    --resize_h=1080
    --resize_w=1920
    --debug=true
```
测试结束后，会将缩放后的图片保存为当前文件夹下dwa_image.bmp。