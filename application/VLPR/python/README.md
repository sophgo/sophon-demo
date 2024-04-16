[简体中文](./README.md)

# python例程

## 目录

- [python例程](#python例程)
  - [目录](#目录)
  - [1. 环境准备](#1-环境准备)
    - [1.1 x86/arm PCIe平台](#11-x86arm-pcie平台)
    - [1.2 SoC平台](#12-soc平台)
  - [3. 推理测试](#3-推理测试)
    - [3.1 参数说明](#31-参数说明)
    - [3.2 运行程序](#32-运行程序)
    - [3.2 程序原理流程图](#32-程序原理流程图)

python目录下提供了python例程以供参考使用，具体情况如下：
| 序号 | python例程 | 说明                                 |
| ---- | ---------- | ------------------------------------ |
| 1    | vlpr.py    | 使用opencv解码、BMCV前处理、BMRT推理 |
| 2    | chars.py   | lprnet后处理使用的汉字字典           |


## 1. 环境准备
### 1.1 x86/arm PCIe平台
如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），可以直接使用它作为开发环境和运行环境。您需要安装libsophon、sophon-opencv、sophon-ffmpeg、sophon-sail，具体步骤可参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。

注意：由于本例程中所用sophon-sail接口较新，请使用如下命令下载最新sophon-sail，并参考[编译安装sophon-sail](../../../docs/Environment_Install_Guide.md###5.3编译安装sophon-sail)进行安装：

```
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
python3 -m dfss --url=open@sophgo.com:/sophon-demo/VLPR/sophon-sail.zip
unzip sophon-sail.zip 
```

### 1.2 SoC平台
如果您使用SoC平台（如SE、SM系列边缘设备），刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包，可直接使用它作为运行环境。
您还需要安装sophon-sail，具体步骤可参考[编译安装sophon-sail](../../../docs/Environment_Install_Guide.md###5.3编译安装sophon-sail)。

## 3. 推理测试
对于PCIe平台和SoC平台，均可以直接在进行推理测试。

### 3.1 参数说明


| 参数名        | 类型   | 说明                                     |
| ------------- | ------ | ---------------------------------------- |
| max_que_size  | int    | 队列长度，默认为16                       |
| video_nums    | string | 视频测试路数，默认为16                   |
| batch_size    | int    | 为输入bmodel的batch_size，默认为4        |
| loops         | int    | 对于一个进程的循环测试图片数，默认为1000 |
| input         | string | 本地视频路径或视频流地址                 |
| yolo_bmodel   | int    | yolov5 bmodel路径                        |
| lprnet_bmodel | int    | lprnet bmodel路径                        |
| dev_id        | int    | 使用的设备id，默认为0号设备              |
| draw_images   | bool   | 是否保存图片，默认为False                |
| stress_test   | bool   | 是否循环压测，默认为False                |

### 3.2 运行程序
运行应用程序即可
```bash
python3 vlpr.py --input ../datasets/1080_1920_30s_512kb.mp4   --loops 1000 --video_nums 16 \
    --yolo_bmodel ../models/yolov5s-licensePlate/BM1684/yolov5s_v6.1_license_3output_int8_4b.bmodel \
    --lprnet_bmodel ../models/lprnet/BM1684/lprnet_int8_4b.bmodel
```
测试过程会打印被检测和识别到的有效车牌信息，测试结束后，会在log中打印FPS等信息。


### 3.2 程序原理流程图
[flow-diagram](../pics/python_pipeline.png)