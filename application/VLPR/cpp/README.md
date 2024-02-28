[简体中文](./README.md)

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
    * [3.2 运行程序](#32-运行程序)
    * [3.3 程序原理流程图](#33-程序原理流程图)

cpp目录下提供了C++例程以供参考使用，具体情况如下：
| 序号  | C++例程      | 说明                                 |
| ---- | ------------ | -----------------------------------  |
| 1    | vlpr_bmcv  | 使用opencv解码、BMCV前处理、BMRT推理   |


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
cd cpp/vlpr_bmcv
mkdir build && cd build
cmake .. 
make
cd ..
```
编译完成后，会在cpp/vlpr_bmcv目录下生成vlpr_bmcv.pcie。

### 2.2 SoC平台
通常在x86主机上交叉编译程序，您需要在x86主机上使用SOPHON SDK搭建交叉编译环境，将程序所依赖的头文件和库文件打包至soc-sdk目录中，具体请参考[交叉编译环境搭建](../../../docs/Environment_Install_Guide.md#41-交叉编译环境搭建)。本例程主要依赖libsophon、sophon-opencv和sophon-ffmpeg运行库包。

交叉编译环境搭建好后，使用交叉编译工具链编译生成可执行文件：
```bash
cd cpp/vlpr_bmcv
mkdir build && cd build
#请根据实际情况修改-DSDK的路径，需使用绝对路径。
cmake -DTARGET_ARCH=soc -DSDK=/path_to_sdk/soc-sdk ..  
make
```
编译完成后，会在cpp/vlpr_bmcv目录下生成vlpr_bmcv.soc。

## 3. 推理测试
对于PCIe平台，可以直接在PCIe平台上推理测试；对于SoC平台，需将交叉编译生成的可执行文件及所需的模型、测试数据拷贝到SoC平台中测试。测试的参数及运行方式是一致的，下面主要以PCIe模式进行介绍。

### 3.1 参数说明
本例程通过读取configs/config.json来配置参数。json格式如下：

```json
{
  "dev_id": 0,
  "yolov5_bmodel_path": "../../models/yolov5s-licensePLate/BM1684X/yolov5s_v6.1_license_3output_int8_1b.bmodel",
  "lprnet_bmodel_path": "../../models/lprnet/BM1684X/lprnet_int8_1b.bmodel",
  "channels": [
    {
      "url": "../../datasets/licenseplate_640516-h264.mp4",
      "is_video": true
    },
    {
      "url": "../../datasets/licenseplate_640516-h264.mp4",
      "is_video": true
    }
  ],
  "yolov5_num_pre": 8,
  "yolov5_num_post": 8,
  "lprnet_num_pre": 8,
  "lprnet_num_post": 8,
  "yolov5_queue_size": 50,
  "lprnet_queue_size": 200,
  "yolov5_conf_thresh": 0.6,
  "yolov5_nms_thresh": 0.6,
  
  "frame_sample_interval": 3,
  "in_frame_num": 5, 
  "out_frame_num": 3, 
  "crop_thread_num": 2, 
  "push_data_thread_num": 2,
  "perf_out_time_interval": 30
}
```
|   参数名               | 类型    |                               说明                       |
|------------------------|---------|-------------------------------------------------------- |
| dev_id                 | int     |                       设备号                             |
| yolov5_bmodel_path     | string  |                 yolov5 bmodel路径                        |
| lprnet_bmodel_path     | string  |                 lprnet bmodel路径                        |
| channels               | list    |                  多路数据地址设置                         |
| url                    | string  |      图片目录路径、本地视频路径或视频流地址                 |
| is_video               | bool    |                    是否是视频格式                         |
| yolov5_num_pre         | int     |   yolov5预处理线程个数，建议不要超过8，需要根据运行结果设置  |
| yolov5_num_post        | int     |   yolov5后处理线程个数，建议不要超过8，需要根据运行结果设置  |
| lprnet_num_pre         | int     |   lprnet预处理线程个数，建议不要超过8，需要根据运行结果设置  |
| lprnet_num_post        | int     |   lprnet后处理线程个数，建议不要超过8，需要根据运行结果设置  |
| yolov5_queue_size      | int     | yolov5线程之间缓存队列的长度，过小会影响性能，需根据结果设置 |
| lprnet_queue_size      | int     | lprnet线程之间缓存队列的长度，过小会影响性能，需根据结果设置 |
| yolov5_conf_thresh     | float   |                  yolov5后处理中置信度阈值                 |
| yolov5_nms_thresh      | float   |                 yolov5后处理中nms置信度阈值               |
| frame_sample_interval  | int     | 跳帧的数量，被跳过的帧会丢弃，由于解码性能原因需要根据结果设置|
| in_frame_num           | int     |           在逻辑处理中，设置多少帧判断为“进”的行为          |
| out_frame_num          | int     |           在逻辑处理中，设置多少帧判断为“出”的行为          |
| crop_thread_num        | int     | 在得到yolov5的检测框后，裁剪车牌函数的线程数，需根据结果设置  |
| push_data_thread_num   | int     | 传递数据给逻辑代码的线程数，建议设置小一点，该线程一般不是瓶颈 |
| perf_out_time_interval | int     | 运行时性能数据输出的时间间隔，以秒为单位，需参考3.2运行程序中打开宏 |


### 3.2 运行程序
配置好json后，运行应用程序即可
```bash
# 对于BM1684X pcie运行
./vlpr_bmcv.pcie --config_path=configs/config_se7.json
# 对于BM1684X soc运行
./vlpr_bmcv.soc --config_path=configs/config_se7.json
# 对于BM1684 pcie运行
./vlpr_bmcv.pcie --config_path=configs/config_se5.json
# 对于BM1684 soc运行
./vlpr_bmcv.soc --config_path=configs/config_se5.json
# 对于SE9-16(BM1688) soc运行
./vlpr_bmcv.soc --config_path=configs/config_se9-16.json
# 对于SE9-8(BM1688) soc运行
./vlpr_bmcv.soc --config_path=configs/config_se9-8.json
```
测试过程会打印被检测和识别到的有效车牌信息，测试结束后，会打印fps等信息。若打开检测可视化的宏开关，即在文件[vlpr_bmcv.cpp](./vlpr_bmcv/vlpr_bmcv.cpp)中设置`#define DET_VIS 1`，会将检测的结果保存在`vis/`目录下。

若打开运行时性能数据输出的宏开关，即在文件[vlpr_bmcv.cpp](./vlpr_bmcv/vlpr_bmcv.cpp)中设置`#define RUNTIME_PERFORMANCE 1`，会实时打印配置参数`perf_out_time_interval`指定的时间间隔内的fps。

若要进行压测，在文件[yolov5.cpp](./vlpr_bmcv/yolov5_multi/yolov5.cpp)中设置`#define PRESSURE 1`，会不断循环视频文件。

> **测试说明**：  
> 1. 出现打印信息"xxx pipe full"，若大部分是前处理流程或推理流程则是正常的，若出现后处理流程，可查看最后一个输出该信息的流程，尝试增加该流程后面一个流程的线程数，即后面一个流程是瓶颈点
> 2. 若结果输出被打印信息"xxx pipe full"覆盖，可设置[datapipe.hpp](vlpr_bmcv/lprnet_multi/datapipe.hpp)中的宏```PIPE_INFO```为0，来关闭打印信息"xxx pipe full"

### 3.3 程序原理流程图
可参考[C++程序原理流程图](../pics/cpp_pipeline.png)，其中yolo部分可参考[C++程序YOLOv5_multi流程图](../../YOLOv5_multi/pics/diagram.png)。
