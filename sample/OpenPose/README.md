# OpenPose
## 目录
* [OpenPose](#openpose)
  * [目录](#目录)
  * [1. 简介](#1-简介)
  * [2. 特性](#2-特性)
  * [3. 准备模型与数据](#3-准备模型与数据)
  * [4. 模型编译](#4-模型编译)
    * [4.1 生成FP32 BModel](#41-生成fp32-bmodel)
    * [4.2 生成INT8 BModel](#42-生成int8-bmodel)
  * [5. 例程测试](#5-例程测试)
  * [6. 精度测试](#6-精度测试)
    * [6.1 测试方法](#61-测试方法)
    * [6.2 测试结果](#62-测试结果)
  * [7. 性能测试](#7-性能测试)
    * [7.1 bmrt_test](#71-bmrt_test)
    * [7.2 程序运行性能](#72-程序运行性能)

## 1. 简介

OpenPose人体姿态识别项目是美国卡耐基梅隆大学（CMU）基于卷积神经网络和监督学习并以caffe为框架开发的开源库。可以实现人体动作、面部表情、手指运动等姿态估计，适用于单人和多人，具有极好的鲁棒性，是世界上首个基于深度学习的实时多人二维姿态估计应用。人体姿态估计技术在体育健身、动作采集、3D试衣、舆情监测等领域具有广阔的应用前景。

**论文:** [OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields](https://arxiv.org/pdf/1611.08050.pdf)

![avatar](pics/pose_face_hands.gif)

本例程对[openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)的人体关键点模型和算法进行移植，使之能在SOPHON BM1684和BM1684X上进行推理测试。

## 2. 特性
* 支持18和25个身体关键点检测
* 支持BM1684X(x86 PCIe、SoC)和BM1684(x86 PCIe、SoC)
* 支持FP32、INT8模型编译和推理
* 支持基于BMCV预处理的C++推理
* 支持基于OpenCV的Python推理
* 支持单batch和多batch模型推理
* 支持图片和视频测试

## 3. 准备模型与数据
您需要准备用于测试的模型和数据集，如果量化模型，还要准备用于量化的数据集。

本例程在`scripts`目录下提供了相关模型和数据集的下载脚本`download.sh`，您也可以自己准备模型和数据集，并参考[4. 模型转换](#4-模型转换)进行模型转换。
```bash
chmod -R +x scripts/
./scripts/download.sh
```
执行后，下载的模型包括：
```
./models
├── BM1684
│   ├── pose_coco_fp32_1b.bmodel              # 用于BM1684的FP32 BModel，batch_size=1，18个身体关键点识别
│   ├── pose_coco_int8_1b.bmodel              # 用于BM1684的INT8 BModel，batch_size=1，18个身体关键点识别
│   ├── pose_coco_int8_4b.bmodel              # 用于BM1684的INT8 BModel，batch_size=4，18个身体关键点识别
│   └── pose_body_25_fp32_1b.bmodel           # 用于BM1684的FP32 BModel，batch_size=1，25个身体关键点识别
├── BM1684X
│   ├── pose_coco_fp32_1b.bmodel              # 用于BM1684X的FP32 BModel，batch_size=1，18个身体关键点识别
│   ├── pose_coco_int8_1b.bmodel              # 用于BM1684X的INT8 BModel，batch_size=1，18个身体关键点识别
│   ├── pose_coco_int8_4b.bmodel              # 用于BM1684X的INT8 BModel，batch_size=4，18个身体关键点识别
│   └── pose_body_25_fp32_1b.bmodel           # 用于BM1684X的FP32 BModel，batch_size=1，25个身体关键点识别
└── caffe/pose
    ├── coco
    │   ├── pose_iter_440000.caffemodel       # 基于COCO的18个身体关键点识别原始模型
    │   └── pose_deploy_linevec.prototxt      # 基于COCO的18个身体关键点识别原始网络配置文件
    └── body_25
        ├── pose_iter_584000.caffemodel       # 25个身体关键点识别原始模型
        └── pose_deploy.prototxt              # 25个身体关键点识别原始网络配置文件
```
下载的数据包括：
```
./datasets
├── test                                      # 测试图片
├── dance_1080P.mp4                           # 测试视频
├── coco128                                   # coco128数据集，用于模型量化
└── coco                                      
    ├── val2017                               # coco val2017数据集
    └── person_keypoints_val2017.json         # coco val2017数据集关键点标签文件，用于计算精度评价指标
```

## 4. 模型编译

caffe原始模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。

模型编译前需要安装TPU-NNTC，具体可参考[tpu-nntc环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-nntc环境搭建)。安装好后需在tpu-nntc环境中进入例程目录。

### 4.1 生成FP32 BModel

caffe模型编译为FP32 BModel，具体方法可参考《TPU-NNTC开发参考手册》的“BMNETC 使用”(请从[算能官网](https://developer.sophgo.com/site/index/material/28/all.html)相应版本的SDK中获取)。

本例程在`scripts`目录下提供了编译FP32 BModel的脚本。请注意修改`gen_fp32bmodel.sh`中的模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（支持BM1684和BM1684X），如：

```bash
./scripts/gen_fp32bmodel.sh BM1684
```

执行上述命令会在`models/BM1684/`下生成`pose_coco_fp32_1b.bmodel`等文件，即转换好的FP32 BModel。

### 4.2 生成INT8 BModel

​不量化模型可跳过本节。

模型的量化方法可参考《TPU-NNTC开发参考手册》的“模型量化”(请从[算能官网](https://developer.sophgo.com/site/index/material/28/all.html)相应版本的SDK中获取)，以及[模型量化注意事项](../../docs/Calibration_Guide.md#1-注意事项)。

​本例程在`scripts`目录下提供了量化INT8 BModel的脚本。请注意修改`gen_int8bmodel.sh`中的模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台（支持BM1684和BM1684X），如：

```shell
./scripts/gen_int8bmodel.sh BM1684
```

​上述脚本会在`models/BM1684`下生成`pose_coco_int8_1b.bmodel`文件，即转换好的INT8 BModel。

## 5. 例程测试
* [C++例程](cpp/README.md)
* [Python例程](python/README.md)

## 6. 精度测试
### 6.1 测试方法

首先，参考[C++例程](cpp/README.md#32-测试图片)或[Python例程](python/README.md#22-测试图片)推理要测试的数据集，生成预测的json文件。  
然后，使用`tools`目录下的`eval_coco.py`脚本，将测试生成的json文件与测试集标签json文件进行对比，计算出目标检测的评价指标，命令如下：
```bash
# 安装pycocotools，若已安装请跳过
pip3 install pycocotools
# 请根据实际情况修改程序路径和json文件路径
python3 tools/eval_coco.py --label_json datasets/coco/person_keypoints_val2017.json --result_json python/results/pose_coco_fp32_1b.bmodel_val2017_opencv_python_result.json
```
### 6.2 测试结果
在coco2017val数据集上，精度测试结果如下：
|   测试平台    |      测试程序       |        测试模型          |AP@IoU=0.5:0.95|AP@IoU=0.5|
| ------------ | ------------------ | -----------------------  | ------------- | -------- |
| BM1684 PCIe  | openpose_opencv.py | pose_coco_fp32_1b.bmodel | 0.408         | 0.669    |
| BM1684 PCIe  | openpose_opencv.py | pose_coco_int8_1b.bmodel | 0.387         | 0.619    |
| BM1684 PCIe  | openpose_bmcv.pcie | pose_coco_fp32_1b.bmodel | 0.395         | 0.677    |
| BM1684 PCIe  | openpose_bmcv.pcie | pose_coco_int8_1b.bmodel | 0.374         | 0.633    |
| BM1684 SoC   | openpose_opencv.py | pose_coco_fp32_1b.bmodel | 0.408         | 0.669    |
| BM1684 SoC   | openpose_opencv.py | pose_coco_int8_1b.bmodel | 0.386         | 0.619    |
| BM1684 SoC   | openpose_bmcv.soc  | pose_coco_fp32_1b.bmodel | 0.395         | 0.676    |
| BM1684 SoC   | openpose_bmcv.soc  | pose_coco_int8_1b.bmodel | 0.374         | 0.633    |
| BM1684X PCIe | openpose_opencv.py | pose_coco_fp32_1b.bmodel | 0.408         | 0.669    |
| BM1684X PCIe | openpose_opencv.py | pose_coco_int8_1b.bmodel | 0.386         | 0.619    |
| BM1684X PCIe | openpose_bmcv.pcie | pose_coco_fp32_1b.bmodel | 0.395         | 0.677    |
| BM1684X PCIe | openpose_bmcv.pcie | pose_coco_int8_1b.bmodel | 0.374         | 0.634    |
| BM1684X SoC  | openpose_opencv.py | pose_coco_fp32_1b.bmodel | 0.408         | 0.669    |
| BM1684X SoC  | openpose_opencv.py | pose_coco_int8_1b.bmodel | 0.386         | 0.619    |
| BM1684X SoC  | openpose_bmcv.soc  | pose_coco_fp32_1b.bmodel | 0.408         | 0.677    |
| BM1684X SoC  | openpose_bmcv.soc  | pose_coco_int8_1b.bmodel | 0.374         | 0.634    |

> **测试说明**：  
1. batch_size=4和batch_size=1的模型精度一致。

## 7. 性能测试
### 7.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径
bmrt_test --bmodel models/BM1684/pose_coco_fp32_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

|            测试模型               | calculate time(ms) |
| -------------------------------  | ------------------ |
| BM1684/pose_coco_fp32_1b.bmodel  | 126                |
| BM1684/pose_coco_int8_1b.bmodel  | 63.1               |
| BM1684/pose_coco_int8_4b.bmodel  | 15.8               |
| BM1684X/pose_coco_fp32_1b.bmodel | 256                |
| BM1684X/pose_coco_int8_1b.bmodel | 9.1                |
| BM1684X/pose_coco_int8_4b.bmodel | 8.9                |

> **测试说明**：  
1. 性能测试结果具有一定的波动性；
2. `calculate time`已折算为平均每张图片的推理时间。

### 7.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++例程打印的预处理时间、推理时间、后处理时间为整个batch处理的时间，需除以相应的batch size才是每张图片的处理时间。

在不同的测试平台上，使用不同的例程、模型测试`datasets/test`，性能测试结果如下：
|    测试平台  |      测试程序       |       测试模型           |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ------------------ | ------------------------ | -------- | --------- | --------- | --------- |
| BM1684 SoC  | openpose_opencv.py | pose_coco_fp32_1b.bmodel | 18.7     | 10.0      | 131       | 5200      |
| BM1684 SoC  | openpose_opencv.py | pose_coco_int8_1b.bmodel | 18.7     | 10.0      | 75        | 5200      |
| BM1684 SoC  | openpose_opencv.py | pose_coco_int8_4b.bmodel | 19.2     | 10.5      | 26.7      | 5200      |
| BM1684 SoC  | openpose_bmcv.soc  | pose_coco_fp32_1b.bmodel | 4.2      | 5.4       | 125       | 527       |
| BM1684 SoC  | openpose_bmcv.soc  | pose_coco_int8_1b.bmodel | 4.2      | 5.4       | 63        | 527       |
| BM1684 SoC  | openpose_bmcv.soc  | pose_coco_int8_4b.bmodel | 4.0      | 5.4       | 15.8      | 527       |
| BM1684X SoC | openpose_opencv.py | pose_coco_fp32_1b.bmodel | 18.5     | 9.4       | 261       | 5270      |
| BM1684X SoC | openpose_opencv.py | pose_coco_int8_1b.bmodel | 18.5     | 9.4       | 20.8      | 5270      |
| BM1684X SoC | openpose_opencv.py | pose_coco_int8_4b.bmodel | 19.1     | 9.9       | 20.0      | 5270      |
| BM1684X SoC | openpose_bmcv.soc  | pose_coco_fp32_1b.bmodel | 3.6      | 0.9       | 256       | 514       |
| BM1684X SoC | openpose_bmcv.soc  | pose_coco_int8_1b.bmodel | 3.6      | 0.8       | 9.0       | 512       |
| BM1684X SoC | openpose_bmcv.soc  | pose_coco_int8_4b.bmodel | 3.4      | 0.8       | 8.9       | 512       |

> **测试说明**：  
1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
3. BM1684/1684X SoC的主控CPU均为8核 ARM A53 42320 DMIPS @2.3GHz，PCIe上的性能由于CPU的不同可能存在较大差异；
4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异。