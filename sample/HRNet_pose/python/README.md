[简体中文](./README.md) | [English](./README_EN.md)

# Python例程

## 目录

- [Python例程](#python例程)
  - [目录](#目录)
  - [1. 环境准备](#1-环境准备)
    - [1.1 x86 PCIe平台](#11-x86-pcie平台)
    - [1.2 SoC平台](#12-soc平台)
  - [2. 推理测试](#2-推理测试)
    - [2.1 参数说明](#21-参数说明)
    - [2.2 测试图片](#22-测试图片)
    - [2.3 测试视频](#23-测试视频)
    - [3. 流程图](#3-流程图)

python/hrnet_pose目录下提供了Python例程，具体情况如下：

| 序号 | Python例程        | 说明                                |
| ---- |-----------------| -----------------------------------  |
| 1    | hrnet_opencv.py | 使用OpenCV解码、OpenCV前处理、SAIL推理 |


## 1. 环境准备
### 1.1 x86 PCIe平台

如果您在x86平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon、sophon-opencv、sophon-ffmpeg和sophon-sail，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)。

此外您还需要配置opencv等其他第三方库：
```bash
pip3 install opencv-python-headless
```

### 1.2 SoC平台

如果您使用SoC平台（如SE、SM系列边缘设备），并使用它测试本例程，刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包。您还需要交叉编译安装sophon-sail，具体可参考[交叉编译安装sophon-sail](../../../docs/Environment_Install_Guide.md#42-交叉编译安装sophon-sail)。

此外您还需要配置opencv等其他第三方库：
```bash
pip3 install opencv-python-headless
```

> **注:**
>
> 上述命令安装的opencv是公版opencv，如果您希望使用sophon-opencv，可以设置如下环境变量：
> ```bash
> export PYTHONPATH=$PYTHONPATH:/opt/sophon/sophon-opencv-latest/opencv-python/
> ```
> **若使用sophon-opencv需要保证python版本小于等于3.8。**

## 2. 推理测试
python例程不需要编译，可以直接运行，PCIe平台和SoC平台的测试参数和运行方式是相同的。
### 2.1 参数说明
hrnet_pose.py的参数说明：
```bash
usage: hrnet_pose.py [-h] [--input INPUT] [--pose_bmodel POSE_BMODEL] [--dev_id DEV_ID]  [--flip FLIP] [--person_thresh PERSON_THRESH] [--detection_bmodel DETECTION_BMODEL] [--conf_thresh CONF_THRESH] [--nms_thresh NMS_THRESH] [--use_cpu_opt] 

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

> **注意：** CPP和python目前都默认关闭nms优化，python调用的优化接口，依赖3.7.0版本之后的sophon-sail，如果您的sophon-sail版本有该接口，可以添加参数`--use_cpu_opt`来开启该接口优化，`use_cpu_opt`仅限输出维度为5的模型(一般是3输出，别的输出个数可能需要用户自行修改后处理代码)。

### 2.2 测试图片
图片测试实例如下，支持对整个图片文件夹进行测试。
```bash
python3 python/hrnet_pose.py --input ./datasets/test_images --pose_bmodel ./models/BM1684X/hrnet_w32_256x192_int8.bmodel --dev_id 0 --flip true --person_thresh 0.5 --detection_bmodel ./models/BM1684X/yolov5s_v6.1_3output_int8_4b.bmodel --conf_thresh 0.01 --nms_thresh 0.6
```
测试结束后，会将预测的图片保存在results/images目录下，同时会打印预测结果、推理时间等信息。
![res](../pics/785_python_opencv.jpg)

### 2.3 测试视频
视频测试实例如下，支持对视频流进行测试。
```bash
python3 python/hrnet_pose.py --input ./datasets/test_pose_estimation.mp4 --pose_bmodel ./models/BM1684X/hrnet_w32_256x192_int8.bmodel --dev_id 0 --flip true --person_thresh 0.5 --detection_bmodel ./models/BM1684X/yolov5s_v6.1_3output_int8_4b.bmodel --conf_thresh 0.01 --nms_thresh 0.6
```
测试结束后，会将预测的结果画在results目录中，与输入文件名相同的.avi文件，同时会打印预测结果、推理时间等信息。

### 3. 流程图

`hrnet_opencv.py`中的处理流程，都遵循以下流程图：

![flowchart](../pics/Flowchart.png)