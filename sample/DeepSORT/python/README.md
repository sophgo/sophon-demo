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
| 1    | deepsort_opencv.py | 使用OpenCV解码、OpenCV前处理、SAIL推理 |

## 1. 环境准备
### 1.1 x86/arm PCIe平台

如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon、sophon-opencv、sophon-ffmpeg和sophon-sail，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。

您需要安装其他第三方库：
```bash
cd python
pip3 install -r requirements.txt
```

### 1.2 SoC平台

如果您使用SoC平台（如SE、SM系列边缘设备），并使用它测试本例程，刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包。您还需要交叉编译安装sophon-sail，具体可参考[交叉编译安装sophon-sail](../../../docs/Environment_Install_Guide.md#42-交叉编译安装sophon-sail)。

您需要安装其他第三方库：
```bash
cd python
pip3 install -r requirements.txt
```

> **注:**
>
> 上述命令安装的是公版opencv，如果您希望使用sophon-opencv，可以设置如下环境变量：
> ```bash
> export PYTHONPATH=$PYTHONPATH:/opt/sophon/sophon-opencv-latest/opencv-python/
> ```
> **若使用sophon-opencv需要保证python版本小于等于3.8。**

## 2. 推理测试
python例程不需要编译，可以直接运行，PCIe平台和SoC平台的测试参数和运行方式是相同的。bmcv例程与opencv运行方式相同，下面以opencv例程为例。

**注意**：本例程依赖YOLOv5例程，在SOC平台上测试时应保证sophon-demo代码仓完整。
### 2.1 参数说明
```bash
usage: deepsort_opencv.py [-h] [--input INPUT] [--bmodel_detector BMODEL_DETECTOR] [--bmodel_extractor BMODEL_EXTRACTOR] [--dev_id DEV_ID]

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT         path of input video or image folder
  --bmodel_detector BMODEL_DETECTOR
                        path of detector bmodel
  --bmodel_extractor BMODEL_EXTRACTOR
                        path of extractor bmodel
  --dev_id DEV_ID       dev id
```
### 2.2 测试MOT数据集
MOT数据集测试实例如下，支持对整个文件夹里的所有图片进行测试。
```bash
cd python
python3 deepsort_opencv.py --input ../datasets/mot15_trainset/ADL-Rundle-6/img1 --bmodel_detector ../models/BM1684X/yolov5s_v6.1_3output_int8_1b.bmodel --bmodel_extractor ../models/BM1684X/extractor_fp32_1b.bmodel --dev_id=0
```
测试结束后，会将预测的图片保存在`results/images`下，预测的结果保存在`results/mot_eval/ADL-Rundle-6_extractor_fp32_1b.bmodel.txt`下，同时会打印推理时间等信息。
```bash
INFO:root:decode_time(ms): 22.19            #平均每帧的解码耗时
INFO:root:encode_time(ms): 64.52            #平均每帧画框和编码的时间
INFO:root:------------------Detector Predict Time Info ----------------------
INFO:root:preprocess_time(ms): 6.12         #目标检测模型平均每帧的预处理耗时
INFO:root:inference_time(ms): 25.93         #目标检测模型平均每帧的推理耗时
INFO:root:postprocess_time(ms): 12.34       #目标检测模型平均每帧的后处理耗时
INFO:root:-------------------------------------------------------------------
INFO:root:------------------Deepsort Tracker Time Info ----------------------
INFO:root:preprocess_time(ms): 0.78         #特征提取模型平均每个crop的预处理耗时
INFO:root:inference_time(ms): 3.17          #特征提取模型平均每个crop的推理耗时
INFO:root:postprocess_time(ms): 13.88       #deepsort平均每帧的的后处理耗时
INFO:root:avg crop num: 8.58                #deepsort平均每帧有多少个crop
INFO:root:-------------------------------------------------------------------
```


### 2.3 测试视频
视频测试实例如下，支持对视频流进行测试。
```bash
cd python
python3 deepsort_opencv.py --input ../datasets/test_car_person_1080P.mp4 --bmodel_detector ../models/BM1684X/yolov5s_v6.1_3output_int8_1b.bmodel --bmodel_extractor ../models/BM1684X/extractor_fp32_1b.bmodel --dev_id=0
```
测试结束后，会将预测的结果画在`results/video/result.mp4`中，同时把图片保存在`results/video`下，预测的结果保存在`results/mot_eval/test_car_person_1080P_extractor_fp32_1b.bmodel.txt`，同时会打印推理时间等信息。  

```bash
INFO:root:decode_time(ms): 2.96             #平均每帧的解码耗时
INFO:root:encode_time(ms): 77.97            #平均每帧画框和编码的时间
INFO:root:------------------Detector Predict Time Info ----------------------
INFO:root:preprocess_time(ms): 7.74         #目标检测模型平均每帧的预处理耗时
INFO:root:inference_time(ms): 26.18         #目标检测模型平均每帧的推理耗时
INFO:root:postprocess_time(ms): 12.83       #目标检测模型平均每帧的后处理耗时
INFO:root:-------------------------------------------------------------------
INFO:root:------------------Deepsort Tracker Time Info ----------------------
INFO:root:preprocess_time(ms): 0.75         #特征提取模型平均每个crop的预处理耗时
INFO:root:inference_time(ms): 3.57          #特征提取模型平均每个crop的推理耗时
INFO:root:postprocess_time(ms): 17.34       #deepsort平均每帧的后处理耗时
INFO:root:avg crop num: 7.13                #deepsort平均每帧有多少个crop
INFO:root:-------------------------------------------------------------------
```
