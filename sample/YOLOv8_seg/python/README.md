# Python例程
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
| 1    | yolov8_opencv.py | 使用OpenCV解码、OpenCV前处理、SAIL推理 |
| 2    | yolov8_bmcv.py   | 使用SAIL解码、BMCV前处理、SAIL推理 |

## 1. 环境准备
### 1.1 x86/arm PCIe平台

如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon、sophon-opencv、sophon-ffmpeg和sophon-sail，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)。或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。

此外您可能还需要安装其他第三方库：
```bash
pip3 install opencv-python-headless
```

### 1.2 SoC平台

如果您使用SoC平台（如SE、SM系列边缘设备），并使用它测试本例程，刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包。您还需要交叉编译安装sophon-sail，具体可参考[交叉编译安装sophon-sail](../../../docs/Environment_Install_Guide.md#42-交叉编译安装sophon-sail)。

此外您可能还需要安装其他第三方库：
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
yolov8_opencv.py和yolov8_bmcv.py的参数一致，以yolov8_opencv.py为例：
```bash
usage: yolov8_opencv.py [--input INPUT_PATH] [--bmodel BMODEL] [--dev_id DEV_ID]
                        [--conf_thresh CONF_THRESH] [--nms_thresh NMS_THRESH]
--input: 测试数据路径，可输入整个图片文件夹的路径或者视频路径；
--bmodel: 用于推理的bmodel路径，默认使用stage 0的网络进行推理；
--dev_id: 用于推理的tpu设备id；
--conf_thresh: 置信度阈值；
--nms_thresh: nms阈值；
--use_tpu_opt: 开启TPU后处理优化；
--getmask_bmodel: TPU后处理优化所需要的getmask bmodel路径。
```

TPU后处理优化的sail接口需要的sophon-sail版本较新，可以用如下命令下载sophon-sail的源码，并参考[sophon-sail python3接口编译安装指南](https://doc.sophgo.com/sdk-docs/v24.04.01/docs_latest_release/docs/sophon-sail/docs/zh/html/1_build.html#python3wheel)自己编译sophon-sail，该例程需要编译包含bmcv,sophon-ffmpeg,sophon-opencv的SAIL。

```bash
pip3 install dfss --upgrade
python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv8_seg/sophon-sail_3.9.0.tar.gz
tar -xvf sophon-sail_3.9.0.tar.gz
```


### 2.2 测试图片
图片测试实例如下，支持对整个图片文件夹进行测试。
```bash
python3 python/yolov8_opencv.py --input datasets/test --bmodel models/BM1688/yolov8s_int8_1b.bmodel --dev_id 0 --conf_thresh 0.25 --nms_thresh 0.7 #--use_tpu_opt
```
测试结束后，会将预测的图片保存在`results/images`下，预测的结果保存在`results/yolov8s_int8_1b.bmodel.bmodel_test_opencv_python_result.json`下，同时会打印预测结果、推理时间等信息。

![res](../pics/3_python_opencv.jpg)

### 2.3 测试视频
视频测试实例如下，支持对视频流进行测试。
```bash
python3 python/yolov8_opencv.py --input datasets/test_car_person_1080P.mp4 --bmodel models/BM1688/yolov8s_int8_1b.bmodel --dev_id 0 --conf_thresh 0.25 --nms_thresh 0.7 #--use_tpu_opt
```
测试结束后，`yolov8_opencv.py`会将预测的结果画在`results/test_car_person_1080P.avi`中，同时会打印预测结果、推理时间等信息。`yolov8_bmcv.py`会将预测结果画在图片上并保存在`results/images`中。
