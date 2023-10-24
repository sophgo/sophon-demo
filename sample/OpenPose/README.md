# OpenPose
## 目录
- [OpenPose](#openpose)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
  - [3. 准备模型与数据](#3-准备模型与数据)
  - [4. 模型编译](#4-模型编译)
  - [5. 例程测试](#5-例程测试)
  - [6. 精度测试](#6-精度测试)
    - [6.1 测试方法](#61-测试方法)
    - [6.2 测试结果](#62-测试结果)
  - [7. 性能测试](#7-性能测试)
    - [7.1 bmrt\_test](#71-bmrt_test)
    - [7.2 程序运行性能](#72-程序运行性能)
  - [8. FAQ](#8-faq)

## 1. 简介

OpenPose人体姿态识别项目是美国卡耐基梅隆大学（CMU）基于卷积神经网络和监督学习并以caffe为框架开发的开源库。可以实现人体动作、面部表情、手指运动等姿态估计，适用于单人和多人，具有极好的鲁棒性，是世界上首个基于深度学习的实时多人二维姿态估计应用。人体姿态估计技术在体育健身、动作采集、3D试衣、舆情监测等领域具有广阔的应用前景。

**论文:** [OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields](https://arxiv.org/pdf/1611.08050.pdf)

![avatar](pics/pose_face_hands.gif)

本例程对[openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)的人体关键点模型和算法进行移植，使之能在SOPHON BM1684和BM1684X上进行推理测试。

## 2. 特性
* 支持18和25个身体关键点检测
* 支持BM1684X(x86 PCIe、SoC)和BM1684(x86 PCIe、SoC、arm PCIe)
* 支持FP32、INT8模型编译和推理
* 支持基于BMCV预处理的C++推理
* 支持基于OpenCV的Python推理
* 支持单batch和多batch模型推理
* 支持图片和视频测试

## 3. 准备模型与数据
您需要准备用于测试的模型和数据集，如果量化模型，还要准备用于量化的数据集。

本例程在`scripts`目录下提供了相关模型和数据集的下载脚本`download.sh`，您也可以自己准备模型和数据集，并参考[4. 模型转换](#4-模型转换)进行模型转换。

```bash
# 安装unzip，若已安装请跳过，非ubuntu系统视情况使用yum或其他方式安装
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```

执行后，模型保存至`models/`，测试数据集下载并解压至`datasets/test/`，精度测试数据集下载并解压至`datasets/coco/val2017_1000/`，量化数据集下载并解压至`datasets/coco128/`

下载的模型包括：
```
./models
├── BM1684
│   ├── pose_coco_fp32_1b.bmodel              # 使用TPU-MLIR编译，用于BM1684的FP32 BModel，batch_size=1，18个身体关键点识别
│   ├── pose_coco_int8_1b.bmodel              # 使用TPU-MLIR编译，用于BM1684的INT8 BModel，batch_size=1，18个身体关键点识别
│   ├── pose_coco_int8_4b.bmodel              # 使用TPU-MLIR编译，用于BM1684的INT8 BModel，batch_size=4，18个身体关键点识别
│   └── pose_body_25_fp32_1b.bmodel           # 使用TPU-MLIR编译，用于BM1684的FP32 BModel，batch_size=1，25个身体关键点识别
├── BM1684X
│   ├── pose_coco_fp32_1b.bmodel              # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=1，18个身体关键点识别
│   ├── pose_coco_fp16_1b.bmodel              # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=1，18个身体关键点识别
│   ├── pose_coco_int8_1b.bmodel              # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=1，18个身体关键点识别
│   ├── pose_coco_int8_4b.bmodel              # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=4，18个身体关键点识别
│   └── pose_body_25_fp32_1b.bmodel           # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=1，25个身体关键点识别
│   └── pose_body_25_fp16_1b.bmodel           # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=1，25个身体关键点识别
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
    ├── val2017_1000                               # coco val2017_1000数据集：coco val2017中随机抽取的1000张样本
    └── person_keypoints_val2017_1000.json         # coco val2017_1000数据集关键点标签文件，用于计算精度评价指标
```

## 4. 模型编译

caffe原始模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。建议使用TPU-MLIR编译BModel。

模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将caffe模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译caffe模型”(请从[算能官网](https://developer.sophgo.com/site/index/material/31/all.html)相应版本的SDK中获取)。

- 生成FP32 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的caffe模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684/BM1684X**），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684
#or
./scripts/gen_fp32bmodel_mlir.sh bm1684x
```

​执行上述命令会在`models/BM1684`或`models/BM1684X/`下生成`pose_body_25_fp32_1b.bmodel`和`pose_coco_fp32_1b.bmodel`文件，即转换好的FP32 BModel。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的caffe模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X**），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x
```

​执行上述命令会在`models/BM1684X/`下生成`pose_body_25_fp16_1b.bmodel`和`pose_coco_fp16_1b.bmodel`文件，即转换好的FP16 BModel。

- 生成INT8 BModel

​本例程在`scripts`目录下提供了量化INT8 BModel的脚本，请注意修改`gen_int8bmodel_mlir.sh`中的caffe模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台（**支持BM1684/BM1684X**），如：

```shell
./scripts/gen_int8bmodel_mlir.sh bm1684
#或
./scripts/gen_int8bmodel_mlir.sh bm1684x
```

​上述脚本会在`models/BM1684`或`models/BM1684X/`下生成`pose_coco_int8_1b.bmodel`等文件，即转换好的INT8 BModel。


## 5. 例程测试
* [C++例程](cpp/README.md)
* [Python例程](python/README.md)

## 6. 精度测试
### 6.1 测试方法

首先，参考[C++例程](cpp/README.md#32-测试图片)或[Python例程](python/README.md#22-测试图片)推理要测试的数据集，生成预测的json文件，注意修改数据集(datasets/coco/val2017_1000)。其中，测试使用的val2017_1000数据集是从coco2017 val数据集中随机抽取1000张样本得到的。然后，使用`tools`目录下的`eval_coco.py`脚本，将测试生成的json文件与测试集标签json文件进行对比，计算出目标检测的评价指标，命令如下：
```bash
# 安装pycocotools，若已安装请跳过
pip3 install pycocotools
# 请根据实际情况修改程序路径和json文件路径
python3 tools/eval_coco.py --gt_path datasets/coco/person_keypoints_val2017_1000.json --result_json python/results/pose_coco_fp32_1b.bmodel_val2017_1000_opencv_python_result.json
```
### 6.2 测试结果
在coco val2017_1000数据集上，精度测试结果如下：
|   测试平台    |      测试程序       |        测试模型          |AP@IoU=0.5:0.95|AP@IoU=0.5|
| ------------ | ------------------ | -----------------------  | ------------- | -------- |
| BM1684 PCIe  | openpose_opencv.py | pose_coco_fp32_1b.bmodel |   0.439  |  0.693 |
| BM1684 PCIe  | openpose_opencv.py | pose_coco_int8_1b.bmodel |   0.433  |  0.692 |
| BM1684 PCIe  | openpose_bmcv.pcie | pose_coco_fp32_1b.bmodel |   0.422  |  0.698 |
| BM1684 PCIe  | openpose_bmcv.pcie | pose_coco_int8_1b.bmodel |   0.413  |  0.688 |
| BM1684 SoC   | openpose_opencv.py | pose_coco_fp32_1b.bmodel |   0.439  |  0.693 |
| BM1684 SoC   | openpose_opencv.py | pose_coco_int8_1b.bmodel |   0.431  |  0.684 |
| BM1684 SoC   | openpose_bmcv.soc  | pose_coco_fp32_1b.bmodel |   0.422  |  0.697 |
| BM1684 SoC   | openpose_bmcv.soc  | pose_coco_int8_1b.bmodel |   0.413  |  0.688 |
| BM1684X PCIe | openpose_opencv.py | pose_coco_fp32_1b.bmodel |   0.439  |  0.693 |
| BM1684X PCIe | openpose_opencv.py | pose_coco_fp16_1b.bmodel |   0.440  |  0.693 |
| BM1684X PCIe | openpose_opencv.py | pose_coco_int8_1b.bmodel |   0.437  |  0.691 |
| BM1684X PCIe | openpose_bmcv.pcie | pose_coco_fp32_1b.bmodel |   0.420  |  0.697 |
| BM1684X PCIe | openpose_bmcv.pcie | pose_coco_fp16_1b.bmodel |   0.420  |  0.697 |
| BM1684X PCIe | openpose_bmcv.pcie | pose_coco_int8_1b.bmodel |   0.418  |  0.697 |
| BM1684X SoC  | openpose_opencv.py | pose_coco_fp32_1b.bmodel |   0.439  |  0.693 |
| BM1684X SoC  | openpose_opencv.py | pose_coco_fp16_1b.bmodel |   0.439  |  0.693 |
| BM1684X SoC  | openpose_opencv.py | pose_coco_int8_1b.bmodel |   0.436  |  0.691 |
| BM1684X SoC  | openpose_bmcv.soc  | pose_coco_fp32_1b.bmodel |   0.420  |  0.697 |
| BM1684X SoC  | openpose_bmcv.soc  | pose_coco_fp16_1b.bmodel |   0.420  |  0.697 |
| BM1684X SoC  | openpose_bmcv.soc  | pose_coco_int8_1b.bmodel |   0.418  |  0.697 |

若设置`use_tpu_kernel_post`为`true`，在coco val2017_1000数据集上，精度测试结果如下：
|   测试平台    |      测试程序       |        测试模型          |AP@IoU=0.5:0.95|AP@IoU=0.5|
| ------------ | ------------------ | -----------------------  | ------------- | -------- |
| BM1684X PCIe | openpose_bmcv.pcie | pose_coco_fp32_1b.bmodel |  0.419   |  0.696 |
| BM1684X PCIe | openpose_bmcv.pcie | pose_coco_fp16_1b.bmodel |  0.419   |  0.696 |
| BM1684X PCIe | openpose_bmcv.pcie | pose_coco_int8_1b.bmodel |  0.418   |  0.694 |
| BM1684X SoC  | openpose_bmcv.soc  | pose_coco_fp32_1b.bmodel |  0.419   |  0.696 |
| BM1684X SoC  | openpose_bmcv.soc  | pose_coco_fp16_1b.bmodel |  0.419   |  0.696 |
| BM1684X SoC  | openpose_bmcv.soc  | pose_coco_int8_1b.bmodel |  0.418   |  0.694 |

若设置`use_tpu_kernel_post`和`restore_half_img_size`为`true`，在coco val2017_1000数据集上，精度测试结果如下：
|   测试平台    |      测试程序       |        测试模型          |AP@IoU=0.5:0.95|AP@IoU=0.5|
| ------------ | ------------------ | -----------------------  | ------------- | -------- |
| BM1684X PCIe | openpose_bmcv.pcie | pose_coco_fp32_1b.bmodel |  0.389   |  0.665 |
| BM1684X PCIe | openpose_bmcv.pcie | pose_coco_fp16_1b.bmodel |  0.389   |  0.665 |
| BM1684X PCIe | openpose_bmcv.pcie | pose_coco_int8_1b.bmodel |  0.390   |  0.668 |
| BM1684X SoC  | openpose_bmcv.soc  | pose_coco_fp32_1b.bmodel |  0.389   |  0.665 |
| BM1684X SoC  | openpose_bmcv.soc  | pose_coco_fp16_1b.bmodel |  0.389   |  0.665 |
| BM1684X SoC  | openpose_bmcv.soc  | pose_coco_int8_1b.bmodel |  0.390   |  0.668 |
 
> **测试说明**：  
1. batch_size=4和batch_size=1的模型精度一致；
2. AP@IoU=0.5:0.95为area=all对应的指标；
3. 本例程未提供arm PCIe平台测试结果。

## 7. 性能测试
### 7.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684/pose_coco_fp32_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

|            测试模型               | calculate time(ms) |
| -------------------------------  | ------------------ |
| BM1684/pose_coco_fp32_1b.bmodel  |    125        |
| BM1684/pose_coco_int8_1b.bmodel  |    63         |
| BM1684/pose_coco_int8_4b.bmodel  |    16         |
| BM1684X/pose_coco_fp32_1b.bmodel |    252        |
| BM1684X/pose_coco_fp16_1b.bmodel |    19         |
| BM1684X/pose_coco_int8_1b.bmodel |    9.4        |
| BM1684X/pose_coco_int8_4b.bmodel |    9.2        |

> **测试说明**：  
1. 性能测试结果具有一定的波动性；
2. `calculate time`已折算为平均每张图片的推理时间；
3. SoC和PCIe的测试结果基本一致。

### 7.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++例程打印的预处理时间、推理时间、后处理时间为整个batch处理的时间，需除以相应的batch size才是每张图片的处理时间。

在不同的测试平台上，使用不同的例程、模型测试`datasets/coco/val2017_1000`，性能测试结果如下：
|    测试平台  |      测试程序       |       测试模型           |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ------------------ | ------------------------ | -------- | --------- | --------- | --------- |
| BM1684 SoC  | openpose_opencv.py | pose_coco_fp32_1b.bmodel |  13.86  |  8.03  | 130.78  | 3068.47  |
| BM1684 SoC  | openpose_opencv.py | pose_coco_int8_1b.bmodel |  13.95  |  8.20  | 74.49   | 3068.18  |
| BM1684 SoC  | openpose_opencv.py | pose_coco_int8_4b.bmodel |  14.07  |  8.81  | 26.83   | 3052.46  |
| BM1684 SoC  | openpose_bmcv.soc  | pose_coco_fp32_1b.bmodel |  5.27   |  1.24  | 125.56  | 302.45   |
| BM1684 SoC  | openpose_bmcv.soc  | pose_coco_int8_1b.bmodel |  5.24   |  1.25  | 62.99   | 301.48   |
| BM1684 SoC  | openpose_bmcv.soc  | pose_coco_int8_4b.bmodel |  5.18   |  1.29  | 15.77   | 306.28   |
| BM1684X SoC | openpose_opencv.py | pose_coco_fp32_1b.bmodel |  15.02  |  7.26  | 257.63  | 3111.41  |
| BM1684X SoC | openpose_opencv.py | pose_coco_fp16_1b.bmodel |  15.00  |  7.30  | 24.60   | 3111.20  |
| BM1684X SoC | openpose_opencv.py | pose_coco_int8_1b.bmodel |  15.02  |  7.33  | 14.96   | 3111.70  |
| BM1684X SoC | openpose_opencv.py | pose_coco_int8_4b.bmodel |  14.99  |  7.42  | 14.22   | 3111.17  |
| BM1684X SoC | openpose_bmcv.soc  | pose_coco_fp32_1b.bmodel |  4.81   |  0.45  | 252.15  | 295.07   |
| BM1684X SoC | openpose_bmcv.soc  | pose_coco_fp16_1b.bmodel |  4.76   |  0.45  | 19.02   | 300.03   |
| BM1684X SoC | openpose_bmcv.soc  | pose_coco_int8_1b.bmodel |  4.76   |  0.45  | 9.37    | 293.81   |
| BM1684X SoC | openpose_bmcv.soc  | pose_coco_int8_4b.bmodel |  4.67   |  0.43  | 9.25    | 296.2    |

若设置`use_tpu_kernel_post`为`true`，在不同的测试平台上，使用不同的例程、模型测试`datasets/coco/val2017_1000`，性能测试结果如下：
|    测试平台  |      测试程序       |       测试模型           |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ------------------ | ------------------------ | -------- | --------- | --------- | --------- |
| BM1684X SoC | openpose_bmcv.soc  | pose_coco_fp32_1b.bmodel |  4.56   |  0.46  | 252.02   | 51.58   |
| BM1684X SoC | openpose_bmcv.soc  | pose_coco_fp16_1b.bmodel |  4.62   |  0.47  | 19.02    | 50.24   |
| BM1684X SoC | openpose_bmcv.soc  | pose_coco_int8_1b.bmodel |  4.60   |  0.47  | 9.37     | 50.43   |
| BM1684X SoC | openpose_bmcv.soc  | pose_coco_int8_4b.bmodel |  4.47   |  0.42  | 9.28     | 50.38    |

  若设置`use_tpu_kernel_post`和`restore_half_img_size`为`true`，在不同的测试平台上，使用不同的例程、模型测试`datasets/coco/val2017_1000`，性能测试结果如下：
|    测试平台  |      测试程序       |       测试模型           |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ------------------ | ------------------------ | -------- | --------- | --------- | --------- |
| BM1684X SoC | openpose_bmcv.soc  | pose_coco_fp32_1b.bmodel |  4.54   |  0.46  | 252.01  | 10.65   |
| BM1684X SoC | openpose_bmcv.soc  | pose_coco_fp16_1b.bmodel |  4.56   |  0.46  | 19.03   | 10.66   |
| BM1684X SoC | openpose_bmcv.soc  | pose_coco_int8_1b.bmodel |  4.60   |  0.46  | 9.37    | 10.43   |
| BM1684X SoC | openpose_bmcv.soc  | pose_coco_int8_4b.bmodel |  4.50   |  0.42  | 9.27    | 10.69   |

> **测试说明**：  
1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
3. BM1684/1684X SoC的主控CPU均为8核 ARM A53 42320 DMIPS @2.3GHz，PCIe上的性能由于CPU的不同可能存在较大差异；
4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异；

## 8. FAQ
请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。