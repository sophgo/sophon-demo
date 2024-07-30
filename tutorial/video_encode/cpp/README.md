# C++例程

## 目录

- [C++例程](#c例程)
  - [目录](#目录)
  - [1. 环境准备](#1-环境准备)
    - [1.1 x86/arm PCIe平台](#11-x86arm-pcie平台)
    - [1.2 SoC平台](#12-soc平台)
  - [2. 程序编译](#2-程序编译)
    - [2.1 x86/arm PCIe平台](#21-x86arm-pcie平台)
      - [2.1.1 bmcv](#211-bmcv)
      - [2.1.2 sail](#212-sail)
    - [2.2 SoC平台](#22-soc平台)
      - [2.2.1 bmcv](#221-bmcv)
      - [2.2.2 sail](#222-sail)
  - [3. 测试](#3-测试)
    - [3.1 参数说明](#31-参数说明)
    - [3.2 测试视频](#32-测试视频)

cpp目录下提供了C++例程以供参考使用，具体情况如下：
| 序号  | C++例程      | 说明                                 |
| ---- | ------------- | -----------------------------------  |
| 1    | [encode_bmcv](./encode_bmcv) | 使用bmcv做编码 |
| 2    | [encode_sail](./encode_sail) | 使用SAIL接口做编码 |

## 1. 环境准备
### 1.1 x86/arm PCIe平台
如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），可以直接使用它作为开发环境和运行环境。您需要安装libsophon、sophon-opencv和sophon-ffmpeg，具体步骤可参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。

### 1.2 SoC平台
如果您使用SoC平台（如SE、SM系列边缘设备），刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包，可直接使用它作为运行环境。通常还需要一台x86主机作为开发环境，用于交叉编译C++程序。


## 2. 程序编译
C++程序运行前需要编译可执行文件。
### 2.1 x86/arm PCIe平台
可以直接在PCIe平台上编译程序：
#### 2.1.1 bmcv
```bash
cd cpp/encode_bmcv
mkdir build && cd build
cmake .. 
make
cd ..
```
编译完成后，会在encode_bmcv目录下生成encode_bmcv.pcie。

#### 2.1.2 sail
如果您使用sophon-sail接口，需要[编译安装sophon-sail](../../../docs/Environment_Install_Guide.md#33-编译安装sophon-sail)，然后进行如下步骤。
```bash
cd cpp/encode_sail
mkdir build && cd build
cmake ..
make
cd ..
```
编译完成后，会在encode_sail目录下生成encode_sail.pcie。

### 2.2 SoC平台
通常在x86主机上交叉编译程序，您需要在x86主机上使用SOPHON SDK搭建交叉编译环境，将程序所依赖的头文件和库文件打包至soc-sdk目录中，具体请参考[交叉编译环境搭建](../../../docs/Environment_Install_Guide.md#41-交叉编译环境搭建)。本例程主要依赖libsophon、sophon-opencv和sophon-ffmpeg运行库包。

交叉编译环境搭建好后，使用交叉编译工具链编译生成可执行文件：
#### 2.2.1 bmcv
```bash
cd cpp/encode_bmcv
mkdir build && cd build
#请根据实际情况修改-DSDK的路径，需使用绝对路径。
cmake -DTARGET_ARCH=soc -DSDK=/path_to_sdk/soc-sdk ..  
make
```
编译完成后，会在encode_bmcv目录下生成encode_bmcv.soc。

#### 2.2.2 sail
如果您使用sophon-sail接口，需要参考[交叉编译安装sophon-sail](../../../docs/Environment_Install_Guide.md#42-交叉编译安装sophon-sail)，给soc环境配置sophon-sail，然后进行如下步骤。
```bash
cd cpp/encode_sail
mkdir build && cd build
#请根据实际情况修改-DSDK和-DSAIL_PATH的路径，需使用绝对路径。
cmake -DTARGET_ARCH=soc -DSDK=/path_to_sdk/soc-sdk -DSAIL_PATH=/path_to_sail/sophon-sail/build_soc/sophon-sail ..
make
```
编译完成后，会在encode_sail目录下生成encode_sail.soc。

## 3. 测试
对于PCIe平台，可以直接在PCIe平台上测试；对于SoC平台，需将交叉编译生成的可执行文件及所需的模型、测试数据拷贝到SoC平台中测试。测试的参数及运行方式是一致的，下面主要以PCIe模式进行介绍。

### 3.1 参数说明
可执行程序默认有一套参数，请注意根据实际情况进行传参，`encode_bmcv.pcie与encode_sail.pcie参数相同。`以encode_bmcv.pcie为例，具体参数说明如下：
```bash
Usage: encode_bmcv.pcie [params]

        --bitrate (value:2000)
                encoded bitrate
        --compressed_nv12 (value:true)
                Whether the format of decoded output is compressed NV12.
        --dev_id (value:0)
                Device id
        --enc_fmt (value:h264_bm)
                encoded video format, h264_bm/hevc_bm
        --framerate (value:25)
                encode frame rate
        --gop (value:32)
                gop size
        --gop_preset (value:2)
                gop_preset
        --height (value:1080)
                The height of the encoded video
        --help (value:true)
                print help information.
        --input_path (value:../datasets/test_car_person_1080P.mp4)
                Path or rtsp url to the video/image file.
        --output_path (value:output.mp4)
                Local file path or stream url
        --pix_fmt (value:NV12)
                encoded pixel format
        --width (value:1920)
                The width of the encoded video

```
### 3.2 测试视频
视频测试实例1如下：
```bash
./encode_bmcv.pcie --input_path=rtsp://127.0.0.1:8554/0 --output_path=test.mp4
```
测试完成后，会将接受的rtsp流保存为test.mp4文件。

视频测试实例2如下：
```bash
./encode_bmcv.pcie --input_path=rtsp://127.0.0.1:8554/0 --output_path=rtsp://127.0.0.1:8554/1
```
测试开始后，会将接受的rtsp流转发到rtsp://127.0.0.1:8554/1。
