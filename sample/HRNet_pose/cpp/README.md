[简体中文](./README.md) | [English](./README_EN.md)

# C++例程

## 目录

- [C++例程](#c++例程)
  - [目录](#目录)
  - [1. 环境准备](#1-环境准备)
    - [1.1 x86 PCIe平台](#11-x86-pcie平台)
    - [1.2 SoC平台](#12-soc平台)
  - [2. 程序编译](#2-程序编译)
    - [2.1 x86 PCIe平台](#21-x86-pcie平台)
      - [2.1.1 bmcv](#211-bmcv)
    - [2.2 SoC平台](#22-soc平台)
      - [2.2.1 bmcv](#221-bmcv)
  - [3. 推理测试](#3-推理测试)
    - [3.1 参数说明](#31-参数说明)
    - [3.2 测试图片](#32-测试图片)
    - [3.3 测试视频](#33-测试视频)
  - [4. 流程图](#4-流程图)


cpp目录下提供了C++例程以供参考使用，具体情况如下：

| 序号 | C++例程        | 说明                               |
| -- | --------------- | ---------------------------------- |
| 1  | hrnet_pose_bmcv | 使用FFmpeg解码、BMCV前处理、BMRT推理 |

## 1. 环境准备

### 1.1 x86 PCIe平台

如果您在x86平台安装了PCIe加速卡（如SC系列加速卡），可以直接使用它作为开发环境和运行环境。您需要安装libsophon、sophon-opencv和sophon-ffmpeg，具体步骤可参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)。

### 1.2 SoC平台

如果您使用SoC平台（如SE、SM系列边缘设备），刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包，可直接使用它作为运行环境。通常还需要一台x86主机作为开发环境，用于交叉编译C++程序。

## 2. 程序编译

C++程序运行前需要编译可执行文件。

### 2.1 x86 PCIe平台

可以直接在PCIe平台上编译程序：

#### 2.1.1 bmcv

```bash
cd cpp/hrnet_pose_bmcv
mkdir build && cd build
cmake .. 
make
cd ..
```

编译完成后，会在hrnet_pose_bmcv目录下生成hrnet_pose_bmcv.pcie。

### 2.2 SoC平台

通常在x86主机上交叉编译程序，您需要在x86主机上使用SOPHON SDK搭建交叉编译环境，将程序所依赖的头文件和库文件打包至soc-sdk目录中，具体请参考[交叉编译环境搭建](../../../docs/Environment_Install_Guide.md#41-交叉编译环境搭建)。本例程主要依赖libsophon、sophon-opencv和sophon-ffmpeg运行库包。

交叉编译环境搭建好后，使用交叉编译工具链编译生成可执行文件：

#### 2.2.1 bmcv

```bash
cd cpp/hrnet_pose_bmcv
mkdir build && cd build
#请根据实际情况修改-DSDK的路径，需使用绝对路径。
cmake -DTARGET_ARCH=soc -DSDK=/path_to_sdk/soc-sdk ..  
make
```

编译完成后，会在hrnet_pose_bmcv目录下生成hrnet_pose_bmcv.soc。如果在SOC模式下执行hrnet_pose_bmcv.soc显示权限不够，可以使用chmod +x ./hrnet_pose_bmcv.soc，给 hrnet_pose_bmcv.soc 文件添加可执行权限。

## 3. 推理测试

对于PCIe平台，可以直接在PCIe平台上推理测试；对于SoC平台，需将交叉编译生成的可执行文件及所需的模型、测试数据拷贝到SoC平台中测试。测试的参数及运行方式是一致的，下面主要以PCIe模式进行介绍。

### 3.1 参数说明

可执行程序默认有一套参数，请注意根据实际情况进行传参。以hrnet_pose_bmcv.pcie为例，具体参数说明如下：

```bash

Usage: hrnet_pose_bmcv.pcie [params] 

optional arguments:
  -h, --help                         打印这个帮助日志然后退出
  --input INPUT                      测试数据路径，可输入整个图片文件夹的路径或者视频路径
  --pose_bmodel POSE_BMODEL          HRNet姿态估计模型的bmodel路径
  --dev_id DEV_ID                    用于推理的tpu设备id
  --flip FLIP                        在推理是是否对图像进行左右翻转以增强姿态估计检测效果
  --person_thresh PERSON_THRESH      目标检测模型对检测到人的置信度阈值
  
  --detection_bmodel DETECTION_BMODEL 目标检测模型的bmodel路径
  --conf_thresh CONF_THRESH          目标检测模型的置信度阈值
  --nms_thresh NMS_THRESH            目标检测模型的nms阈值
  --use_cpu_opt                      开启YOLOv5的cpu后处理优化
```

> **注意：** cpp例程传参与python不同，需要用等于号，例如`./hrnet_pose_bmcv.pcie --input=xxx`。cpp可以使用`--use_cpu_opt=true`开启前置yolov5模型后处理cpu加速，`use_cpu_opt`仅限输出维度为5的模型(一般是3输出，别的输出个数可能需要用户自行修改后处理代码)。

### 3.2 测试图片

图片测试实例如下，支持对整个图片文件夹进行测试。

```bash
./hrnet_pose_bmcv.pcie --input=../../datasets/test_images --pose_bmodel=../../models/BM1684X/hrnet_w32_256x192_int8.bmodel --person_thresh=0.5 --detection_bmodel=../../models/BM1684X/yolov5s_v6.1_3output_int8_4b.bmodel --conf_thresh=0.01 --nms_thresh=0.6 --classnames=../../datasets/coco.names
```

测试结束后，会将预测的图片保存在`results/images`下，预测的结果保存在`results/keypoints_results_cpp.json`下，同时会打印预测结果、推理时间等信息。

![res](../pics/785_cpp_bmcv.jpg)


### 3.3 测试视频

视频测试实例如下，支持对视频流进行测试。

```bash
./hrnet_pose_bmcv.pcie --input=../../datasets/test_pose_estimation.mp4 --pose_bmodel=../../models/BM1684X/hrnet_w32_256x192_int8.bmodel --person_thresh=0.5 --detection_bmodel=../../models/BM1684X/yolov5s_v6.1_3output_int8_4b.bmodel --conf_thresh=0.01 --nms_thresh=0.6 --classnames=../../datasets/coco.names
```

测试结束后，会将预测结果画在图片上并保存在`results/video`中，同时会打印预测结果、推理时间等信息。


### 4. 流程图

`hrnet_pose_bmcv`中的处理流程，都遵循以下流程图：

![flowchart](../pics/Flowchart.png)