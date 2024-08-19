# Python例程

## 目录

- [Python例程](#python例程)
  - [目录](#目录)
  - [1. 环境准备](#1-环境准备)
    - [1.1 x86/arm PCIe平台](#11-x86arm-pcie平台)
    - [1.2 SoC平台](#12-soc平台)
  - [2. 推理测试](#2-推理测试)
    - [2.1 参数说明](#21-参数说明)
    - [2.2 测试视频](#22-测试视频)

python目录下提供了python例程以供参考使用，具体情况如下：
| 序号  |   Python例程     |                                           说明                                                       |
| ---- | ---------------- | --------------------------------------------------------------------------------------------------- |
| 1    | yolov5_bmcv.py   |      一个调用SAIL解码、BMCV前处理、SAIL推理的多路YOLOv5检测例程，第N路视频流会在第N%2个npu core做推理          |

## 1. 环境准备
### 1.1 x86/arm PCIe平台
BM1688暂不支持x86/arm PCIe平台。

### 1.2 SoC平台
如果您使用SoC平台（如SE、SM系列边缘设备），并使用它测试本例程，刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包，可直接使用它作为运行环境。

此外，您还需要安装其他第三方库：
```bash
pip3 install opencv-python opencv-python-headless
```
由于本例程需要的sophon-sail版本较新，这里提供一个可用的sophon-sail whl包，SoC环境下可用以下命令下载和安装：
```bash
pip3 install dfss --upgrade
python3 -m dfss --url=open@sophgo.com:sophon-demo/ChatGLM3/sail/soc/SE9/sophon-3.8.0-py3-none-any.whl #arm soc, py38, for SE9
pip3 install sophon-3.8.0-py3-none-any.whl
```
如果您需要其他python版本的sophon-sail，可以参考[SoC平台交叉编译安装sophon-sail](../../../docs/Environment_Install_Guide.md#42-交叉编译安装sophon-sail)，到官网下载sophon-sail (sail的版本>=v3.8.0,对应BM1684&BM1684X SDK>=V24.04.01, BM1688&CV186AH SDK>=V1.6.0)自己编译。

## 2. 推理测试
python例程不需要编译，可以直接运行。
### 2.1 参数说明
可执行程序默认有一套参数，请注意根据实际情况进行传参，以yolov5_bmcv.py为例，具体参数说明如下：
```bash
Usage: yolov5_bmcv.py [params]

        --bmodel (value:../models/BM1688/yolov5s_v6.1_3output_int8_4b.bmodel)
                bmodel file path
        --chan_num (value:2)
                copy the input video into chan_num copies
        --classnames (value:../datasets/coco.names)
                class names file path
        --conf_thresh (value:0.5)
                confidence threshold for filter boxes
        --dev_id (value:0)
                TPU device id
        --input (value:../datasets/test_car_person_1080P.mp4)
                input video file path
        --nms_thresh (value:0.5)
                iou threshold for nms
```
### 2.2 测试视频
2路测试实例如下：
```bash
python3 yolov5_bmcv.py --chan_num=2
```
测试结束后，会将视频中的帧以jpg图片的形式保存在`results/images_chan_N`中，N表示第N路。

测试过程中，您可以通过如下命令查看BM1688两个npu core的占用情况：
```bash
cat /sys/class/bm-tpu/bm-tpu0/device/npu_usage
```