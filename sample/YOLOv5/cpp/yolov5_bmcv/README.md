

# C++例程
- [C++例程](#c例程)
  - [1. 目录说明](#1-目录说明)
  - [2.程序编译](#2程序编译)
  - [2. 1 PCIE模式](#2-1-pcie模式)
    - [2.1.1 环境](#211-环境)
    - [2.1.2 程序编译](#212-程序编译)
  - [2. 2 SOC模式](#2-2-soc模式)
    - [2.2.1 环境](#221-环境)
    - [2.2.2 程序编译](#222-程序编译)
  - [3. 测试](#3-测试)

## 1. 目录说明

​	cpp目录下提供了C++例程以供参考使用，目录结构如下：

```
yolov5_bmcv
├── bmnn_utils.h
├── bm_wrapper.hpp
├── CMakeLists.txt    # cmake编译脚本
├── main.cpp          # 主程序
├── README.md
├── utils.hpp
├── yolov5.cpp        # YOLOv5实现
└── yolov5.hpp        # YOLOv5头文件

```

## 2.程序编译

## 2. 1 PCIE模式

### 2.1.1 环境

​	硬件：x86平台，并安装了1684X PCIE加速卡

​	软件：libsophon、sophon-opencv、sophon-ffmpeg，可联系技术支持获取

### 2.1.2 程序编译

​	C++程序运行前需要编译可执行文件，命令如下：

```bash
cd cpp/yolov5_bmcv
mkdir build
cd build
rm ./* -rf
cmake -DTARGET_ARCH=x86 ..
make
```

​	运行成功后，会在build上级目录下生成可执行文件，如下：

```
yolov5_bmcv
├──......
└── yolov5_demo.pcie    #可执行程序
```

## 2. 2 SOC模式

### 2.2.1 环境

​	硬件：x86平台(交叉编译)

​	软件：libsophon、sophon-mw、sophon-img，相应成果物请联系技术支持进行获取

​	编译工具链：aarch64-linux-gnu，可参考《LIBSOPHON使用手册》手册中SOC MODE章节说明进行安装（注意lib、include的准备）

### 2.2.2 程序编译

​	C++程序运行前需要编译可执行文件，命令如下：

```bash
cd cpp/yolov5_bmcv
mkdir build
cd build
rm ./* -rf
cmake -DTARGET_ARCH=soc -DSDK={实际soc sdk路径} ..
make
```

​	运行成功后，会在build上级目录下生成可执行文件，如下：

```
yolov5_bmcv
├──......
└── yolov5_demo.soc    #可执行程序
```



## 3. 测试

可执行程序默认有一套参数，，请注意根据实际情况进行传参，具体参数说明如下：

```bash
Usage: yolov5_demo.pcie/soc [params]

        --bmodel (value:../data/models/fp32bmodel/yolov5s_640_coco_v6.1_3output_fp32_1b.bmodel)
                bmodel file path
        --classnames (value:../data/images/coco.names)
                class names' file path
        --conf (value:0.5)
                confidence threshold for filter boxes
        --frame_num (value:0)
                number of frames in video to process, 0 means processing all frames
        --help (value:true)
                Print help information.
        --input (value:../data/images/zidane.jpg)
                input stream file path
        --iou (value:0.5)
                iou threshold for nms
        --is_video (value:0)
                input video file path
        --obj (value:0.5)
                object score threshold for filter boxes
        --tpuid (value:0)
                TPU device id
```

​	demo中支持单图、文件夹、视频测试，按照实际情况传入参数即可，默认是单图。另外，模型支持fp3bmodel、int8bmodel，可以通过传入模型路径参数进行测试：

```bash
# PCIE mode、x86环境下运行，默认BM1684X平台、单图，请根据实际情况传参
./yolov5_demo.pcie  

# SOC mode、BM168X环境下运行，默认BM1684X平台、单图，请根据实际情况传参
./yolov5_demo.soc  
```

注：

1. 程序执行完毕后，会通过终端打印的方式给出各阶段耗时
2. 耗时统计存在略微波动属于正常现象
3. CPP传参与python不同，需要用等于号，例如：`./yolov5_demo.pcie --bmodel=xxx`