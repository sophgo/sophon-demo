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
    * [3.2 测试视频](#32-测试视频)

cpp目录下提供了C++例程以供参考使用，具体情况如下：
| 序号  | C++例程      | 说明                                 |
| ---- | ------------- | -----------------------------------  |
| 1    | yolov5_bmcv | 一个调用sophon-openCV解码、BMCV前处理、BMRT推理的多路YOLOv5检测例程，第N路视频流会在第N%2个npu core做推理 |

## 1. 环境准备
### 1.1 x86/arm PCIe平台
BM1688暂不支持x86/arm PCIe平台。

### 1.2 SoC平台
如果您使用SoC平台（如SE、SM系列边缘设备），刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包，可直接使用它作为运行环境。通常还需要一台x86主机作为开发环境，用于交叉编译C++程序。


## 2. 程序编译
C++程序运行前需要编译可执行文件。
### 2.1 x86/arm PCIe平台
BM1688暂不支持x86/arm PCIe平台。

### 2.2 SoC平台
通常在x86主机上交叉编译程序，您需要在x86主机上使用SOPHON SDK搭建交叉编译环境，将程序所依赖的头文件和库文件打包至soc-sdk目录中，具体请参考[交叉编译环境搭建](../../../docs/Environment_Install_Guide.md#41-交叉编译环境搭建)。本例程主要依赖libsophon、sophon-opencv和sophon-ffmpeg运行库包。

交叉编译环境搭建好后，使用交叉编译工具链编译生成可执行文件：
```bash
mkdir build && cd build
#请根据实际情况修改-DSDK的路径，需使用绝对路径。
cmake -DTARGET_ARCH=soc -DSDK=/path_to_sdk/soc-sdk ..  
make
```
编译完成后，会在当前目录下生成yolov5_bmcv.soc。

## 3. 推理测试
对于SoC平台，需将交叉编译生成的可执行文件及所需的模型、测试数据拷贝到SoC平台中测试。

### 3.1 参数说明
可执行程序默认有一套参数，请注意根据实际情况进行传参，以yolov5_bmcv.soc为例，具体参数说明如下：
```bash
Usage: yolov5_bmcv.soc [params]

        --bmodel (value:../../models/BM1688/yolov5s_v6.1_3output_int8_4b.bmodel)
                bmodel file path
        --chan_num (value:2)
                copy the input video into chan_num copies
        --classnames (value:../../datasets/coco.names)
                class names file path
        --conf_thresh (value:0.5)
                confidence threshold for filter boxes
        --dev_id (value:0)
                TPU device id
        --help (value:true)
                print help information.
        --input (value:../../datasets/test_car_person_1080P.mp4)
                input video file path
        --nms_thresh (value:0.5)
                iou threshold for nms
```
### 3.2 测试视频
2路测试实例如下：
```bash
./yolov5_bmcv.soc --chan_num=2
```
测试结束后，会将视频中的帧以jpg图片的形式保存在`results/images_chan_N`中，N表示第N路。

测试过程中，您可以通过如下命令查看BM1688两个npu core的占用情况：
```bash
cat /sys/class/bm-tpu/bm-tpu0/device/npu_usage
```