# Python例程

## 目录

* [1. 环境准备](#1-环境准备)
    * [1.1 x86/arm PCIe平台](#11-x86arm-pcie平台)
    * [1.2 SoC平台](#12-soc平台)
* [2. 推理测试](#2-推理测试)
    * [2.1 参数说明](#21-参数说明)
    * [2.2 测试图片](#22-测试图片)
    * [2.3 测试视频](#23-测试视频)

python目录下提供了一系列Python例程，具体情况如下：

| 序号 |  Python例程      | 说明                                |
| ---- | ---------------- | -----------------------------------  |
| 1    | [ocv_vidbasic.py](./ocv_vidbasic.py) | 使用sophon-opencv硬件加速实现视频解码，并将视频记录为png或jpg格式 |

## 1. 环境准备
### 1.1 x86/arm PCIe平台

如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon、sophon-opencv和sophon-ffmpeg，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。

### 1.2 SoC平台

如果您使用SoC平台（如SE、SM系列边缘设备），并使用它测试本例程，刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包。

## 2. 测试
python例程不需要编译，只需要导入sophon-opencv的python接口，手动设置环境变量后就可以直接运行，PCIe平台和SoC平台的测试参数和运行方式是相同的。
### 2.1 参数说明
```bash
usage: ocv_vidbasic.py [-h] [--video_path VIDEO_PATH] [--device_id DEVICE_ID] [--frame_num FRAME_NUM] [--width WIDTH]
                       [--height HEIGHT] [--output_name OUTPUT_NAME]

optional arguments:
  -h, --help            show this help message and exit
  --video_path VIDEO_PATH
                        input video path
  --device_id DEVICE_ID
                        device id
  --frame_num FRAME_NUM
                        encode and decode frame number
  --width WIDTH         width of sampler
  --height HEIGHT       height of sampler
  --output_name OUTPUT_NAME
                        output name of frame
```

### 2.2 测试图片
图片测试实例如下
```bash
# 手动设置环境变量，使用sophon-opencv的python接口
export PYTHONPATH=$PYTHONPATH:/opt/sophon/sophon-opencv-latest/opencv-python

python3 ocv_vidbasic.py --video_path road.mp4 --frame_num 300 --device_id 0 --output_name out
```
测试结束后，会将视频解码后再记录为jpg格式，保存到当前文件夹下。
