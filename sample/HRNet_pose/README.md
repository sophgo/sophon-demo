[简体中文](./README.md) | [English](./README_EN.md)

# HRNet_pose

## 目录

- [HRNet_pose](#hrnet_pose)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
    - [2.1 目录结构说明](#21-目录结构说明)
    - [2.2 SDK特性](#22-sdk特性)
    - [2.3 算法特性](#23-算法特性)
  - [3. 数据准备与模型编译](#3-数据准备与模型编译)
    - [3.1 数据准备](#31-数据准备)
    - [3.2 模型编译](#32-模型编译)
  - [4. 例程测试](#4-例程测试)
  - [5. 精度测试](#5-精度测试)
    - [5.1 测试方法](#51-测试方法)
    - [5.2 测试结果](#52-测试结果)
  - [6. 性能测试](#6-性能测试)
    - [6.1 bmrt_test](#61-bmrt_test)
    - [6.2 程序运行性能](#62-程序运行性能)
  

## 1. 简介
HRNet（High-Resolution Net）是针对2D人体姿态估计（Human Pose Estimation或Keypoint Detection）任务提出的，并且该网络主要是针对单一个体的姿态评估（即输入网络的图像中应该只有一个人体目标）。人体姿态估计在现今的应用场景也比较多，比如说人体行为动作识别，人机交互（比如人作出某种动作可以触发系统执行某些任务），动画制作（比如根据人体的关键点信息生成对应卡通人物的动作）等等。本例程对[​HRNet官方开源仓库](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)中的模型和算法进行移植，使之能在SOPHON BM1684X/BM1688/CV186X上进行推理测试。此外，部分代码参考了https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_keypoint/HRNet。

## 2. 特性

### 2.1 目录结构说明
```bash
├── cpp                             # 存放C++例程及其README
|   ├──README_EN.md     
|   ├──README.md      
|   ├──hrnet_pose_bmcv              # 使用FFmpeg解码、BMCV前处理、BMRT推理的C++例程
├── docs                            # 存放本例程专用文档，如ONNX导出、移植常见问题等
├── pics                            # 存放README等说明文档中用到的图片
├── python                          # 存放Python例程及其README
|   ├──README_EN.md 
|   ├──README.md 
|   ├── hrnet_pose 
|       ├──__init__.py  
|       ├──hrnet_opencv.py          # 使用OpenCV解码、OpenCV前处理、SAIL推理的Python例程
|       ├──preprocess_hrnet.py      # hrnet前处理
|       ├──postprocess_hrnet.py     # hrnet后处理
|       ├──utils_hrnet.py           # hrnet所用到的相关常量
|   ├── yolov5                      # yolov5 相关代码，详见 ../YOLOv5
|   ├──hrnet_pose.py                # 使用yolov5做前置检测模型，然后使用hrnet进行人体关键点检测的Python例程
├── README_EN.md                    # 本例程的英文指南
├── README.md                       # 本例程的中文指南
├── scripts                         # 存放模型编译、数据下载、自动测试等shell脚本
└── tools                           # 存放精度测试python脚本
```

### 2.2 SDK特性
* 支持BM1688/CV186X(SoC)、BM1684X(x86 PCIe、SoC)
* 支持FP32、FP16(BM1684X/BM1688/CV186X)、INT8模型编译和推理
* 支持基于BMCV预处理的C++推理
* 支持基于OpenCV预处理的Python推理
* 前置目标检测模型YOLOv5，支持单batch和多batch模型的推理，支持1个输出和3个输出的模型推理
* HRNet模型仅支持单batch推理
* 支持图片和视频测试


## 3. 准备模型与数据

### 3.1 数据准备

​本例程在`scripts`目录下提供了相关模型和数据的下载脚本`download.sh`，**如果您希望自己准备模型和数据集，可以跳过本小节，参考[3.2 模型编译](#32-模型编译)进行模型转换。**

```bash
chmod -R +x scripts/
./scripts/download.sh
```

下载的模型包括：
```
./models
├── BM1684X    
│   ├── hrnet_w32_256x192_f32.bmodel              # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=1
│   ├── hrnet_w32_256x192_f16.bmodel              # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=1
│   ├── hrnet_w32_256x192_int8.bmodel             # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=1
│   ├── yolov5s_v6.1_3output_int8_4b.bmodel      # 使用TPU-MLIR编译，用于BM1684X的前置YOLOv5的INT8 BModel，batch_size=4
├── BM1688
│   ├── hrnet_w32_256x192_f32.bmodel              # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1
│   ├── hrnet_w32_256x192_f16.bmodel              # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1
│   ├── hrnet_w32_256x192_int8.bmodel             # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=1
│   ├── hrnet_w32_256x192_f32_2core.bmodel        # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1，num_core=2
│   ├── hrnet_w32_256x192_f16_2core.bmodel        # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1，num_core=2
│   ├── hrnet_w32_256x192_int8_2core.bmodel       # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=1，num_core=2
│   ├── yolov5s_v6.1_3output_int8_4b.bmodel       # 使用TPU-MLIR编译，用于BM1688的前置YOLOv5的INT8 BModel，batch_size=4
│
├── CV186X
│   ├── hrnet_w32_256x192_f32.bmodel              # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1
│   ├── hrnet_w32_256x192_f16.bmodel              # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1
│   ├── hrnet_w32_256x192_int8.bmodel             # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=1
│   ├── yolov5s_v6.1_3output_int8_4b.bmodel       # 使用TPU-MLIR编译，用于BM1688的前置YOLOv5的INT8 BModel，batch_size=4
└── onnx
    └── pose_hrnet_w32_256x192.onnx               # 相关HRNet模型，此处没有benchmark，用户自行使用。
     
```

下载的数据包括：
```
./datasets
├── test_images                                   # 用于测试的图片
├── test_pose_estimation.mp4                      # 用于测试的视频
├── coco.names                                    # coco类别名文件
├── single_person_images_100                      # 从coco val2017数据集标注信息中截取的单张人体图片，随机选择100张，可用于进行量化时生成校准表
└── coco                                      
    ├── val2017                                   # coco val2017数据集
    └── person_keypoints_val2017.json             # coco val2017数据集对应的人体关键点标签文件，用于计算精度评价指标  
```

其他数据包括：
```
./mlir_utils
├── person.jpg                                    # 生成FP32 BModel FP16 BModel INT8 BModel 时的--test_input参数
└── qtable                                        # 生成INT8 BModel时需要将某些层设置为FP32或者F16，即混合精度量化
 
```

### 3.2 模型编译

**如果您不编译模型，只想直接使用下载的数据集和模型，可以跳过本小节。**

源模型需要编译成BModel才能在SOPHON TPU上运行，源模型在编译前要导出成onnx模型，如果您使用的TPU-MLIR版本>=v1.3.0（即官网v23.07.01），也可以直接使用torchscript模型。​同时，您需要准备用于测试的数据集，如果量化模型，还要准备用于量化的数据集，通常，量化数据集随机从训练数据集中选择，10-100张左右。

建议使用TPU-MLIR编译BModel，模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录，并使用本例程提供的脚本将onnx模型编译为BModel。脚本中命令的详细说明可参考《TPU-MLIR开发手册》(请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的SDK中获取)。

- 生成FP32 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684x #/bm1688/cv186x
```

​执行上述命令会在`models/BM1684X`等文件夹下生成`hrnet_w32_256x192_f32.bmodel`文件，即转换好的FP32 BModel。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x #bm1688/cv186x
```

​执行上述命令会在`models/BM1684X/`等文件夹下生成`hrnet_w32_256x192_f16.bmodel`文件，即转换好的FP16 BModel。

- 生成INT8 BModel

​本例程在`scripts`目录下提供了量化INT8 BModel的脚本，请注意修改`gen_int8bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台（**BM1684X/BM1688/CV186X**），如：

```shell
./scripts/gen_int8bmodel_mlir.sh bm1684x #/bm1688/cv186x
```

​上述脚本会在`models/BM1684X`等文件夹下生成`hrnet_w32_256x192_int8.bmodel`等文件，即转换好的INT8 BModel。

## 4. 例程测试
- [C++例程](./cpp/README.md)
- [Python例程](./python/README.md)


## 5. 精度测试
### 5.1 测试方法

首先，参考[C++例程](cpp/README.md#32-测试图片)或[Python例程](python/README.md#22-测试图片)推理要测试的数据集，生成预测的json文件，注意修改数据集(datasets/coco/val2017)和相关参数(person_thresh=0.5，conf_thresh=0.01，nms_thresh=0.6, flip=true)。  
然后，使用`tools`目录下的`eval_coco.py`脚本，将测试生成的json文件与测试集标签json文件进行对比，计算出人体关键点检测的评价指标mAP，命令如下：

```bash
# 安装pycocotools，若已安装请跳过
pip3 install pycocotools
# 请根据实际情况修改测试结果路径和标注文件路径
python3 tools/eval_coco.py --results_path=./results/keypoints_results_python.json --gt_path=./datasets/coco/person_keypoints_val2017.json
```
### 5.2 测试结果
python和C++不设置`--use_cpu_opt`进行测试，在`datasets/coco/val2017`数据集上，使用目标检测模型yolov5s_v6.1_3output_int8_4b.bmodel，精度测试结果如下：

|   测试平台      |      测试程序      |              测试模型                            | AP@IoU=0.5:0.95  | AP@IoU=0.5    |
| ---------------- | ------------------------ | ----------------------------------------|------------------|---------------|
| SE7-32           | hrnet_pose.py            |  hrnet_w32_256x192_fp32.bmodel          |    0.596    |         0.736      |
| SE7-32           | hrnet_pose.py            |  hrnet_w32_256x192_fp16.bmodel          |    0.596    |         0.736      |
| SE7-32           | hrnet_pose.py            |  hrnet_w32_256x192_int8.bmodel          |    0.593    |         0.736      |
| SE7-32           | hrnet_pose_bmcv.soc      |  hrnet_w32_256x192_fp32.bmodel          |    0.622    |         0.746      |
| SE7-32           | hrnet_pose_bmcv.soc      |  hrnet_w32_256x192_fp16.bmodel          |    0.623    |         0.746      |
| SE7-32           | hrnet_pose_bmcv.soc      |  hrnet_w32_256x192_int8.bmodel          |    0.621    |         0.746      |
| SE9-16           | hrnet_pose.py            |  hrnet_w32_256x192_fp32.bmodel          |    0.599         |    0.736      |
| SE9-16           | hrnet_pose.py            |  hrnet_w32_256x192_fp16.bmodel          |    0.599         |    0.736      |
| SE9-16           | hrnet_pose.py            |  hrnet_w32_256x192_int8.bmodel          |    0.598         |    0.736      |
| SE9-16           | hrnet_pose_bmcv.soc      |  hrnet_w32_256x192_fp32.bmodel          |    0.626         |    0.747      |
| SE9-16           | hrnet_pose_bmcv.soc      |  hrnet_w32_256x192_fp16.bmodel          |    0.626         |    0.746      |
| SE9-16           | hrnet_pose_bmcv.soc      |  hrnet_w32_256x192_int8.bmodel          |    0.624         |    0.746      |
| SE9-16           | hrnet_pose.py            |  hrnet_w32_256x192_fp32_2core.bmodel    |    0.599    |         0.736      |
| SE9-16           | hrnet_pose.py            |  hrnet_w32_256x192_fp16_2core.bmodel    |    0.599    |         0.736      |
| SE9-16           | hrnet_pose.py            |  hrnet_w32_256x192_int8_2core.bmodel    |    0.598    |         0.736      |
| SE9-16           | hrnet_pose_bmcv.soc      |  hrnet_w32_256x192_fp32_2core.bmodel    |    0.627    |         0.747      |
| SE9-16           | hrnet_pose_bmcv.soc      |  hrnet_w32_256x192_fp16_2core.bmodel    |    0.627    |         0.746      |
| SE9-16           | hrnet_pose_bmcv.soc      |  hrnet_w32_256x192_int8_2core.bmodel    |    0.624    |         0.746      |
| SE9-8            | hrnet_pose.py            |  hrnet_w32_256x192_fp32.bmodel          |    0.599    |         0.736      |
| SE9-8            | hrnet_pose.py            |  hrnet_w32_256x192_fp16.bmodel          |    0.599    |         0.736      |
| SE9-8            | hrnet_pose.py            |  hrnet_w32_256x192_int8.bmodel          |    0.598    |         0.736      |
| SE9-8            | hrnet_pose_bmcv.soc      |  hrnet_w32_256x192_fp32.bmodel          |    0.627    |         0.746      |
| SE9-8            | hrnet_pose_bmcv.soc      |  hrnet_w32_256x192_fp16.bmodel          |    0.627    |         0.746      |
| SE9-8            | hrnet_pose_bmcv.soc      |  hrnet_w32_256x192_int8.bmodel          |    0.624    |         0.746      |


> **测试说明**：  
> 1. 由于sdk版本之间可能存在差异，实际运行结果与本表有<0.01的精度误差是正常的；
> 2. AP@IoU=0.5:0.95为area=all对应的指标；
> 3. 在搭载了相同TPU和SOPHONSDK的PCIe或SoC平台上，相同程序的精度一致，SE7系列对应BM1684X，SE9系列中，SE9-16对应BM1688，SE9-8对应CV186X；

## 6. 性能测试
### 6.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径
bmrt_test --bmodel models/BM1684X/hrnet_w32_256x192_f32.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

|                  测试模型                         | calculate time(ms) |
| ------------------------------------------------- | ----------------- |
| BM1684X/hrnet_w32_256x192_f32.bmodel        |            16.0 |
| BM1684X/hrnet_w32_256x192_f16.bmodel        |            1.90 |
| BM1684X/hrnet_w32_256x192_int8.bmodel       |            1.33 |
| BM1688/hrnet_w32_256x192_f32.bmodel         |            76.4 |
| BM1688/hrnet_w32_256x192_f16.bmodel         |            9.83 |
| BM1688/hrnet_w32_256x192_int8.bmodel        |            3.32 |
| BM1688/hrnet_w32_256x192_f32_2core.bmodel   |            58.9 |
| BM1688/hrnet_w32_256x192_f16_2core.bmodel   |            9.59 |
| BM1688/hrnet_w32_256x192_int8_2core.bmodel  |            3.20 |
| CV186X/hrnet_w32_256x192_f32.bmodel         |            76.4 |
| CV186X/hrnet_w32_256x192_f16.bmodel         |            9.92 |
| CV186X/hrnet_w32_256x192_int8.bmodel        |            3.43 |

> **测试说明**：  
> 1. 性能测试结果具有一定的波动性；
> 2. `calculate time`已折算为平均每张图片的推理时间；
> 3. SoC和PCIe的测试结果基本一致。

### 6.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间和后处理时间。C++和Python例程打印的时间已经折算为单张图片的处理时间。

CPP设置`--use_cpu_opt=false`或python不设置`--use_cpu_opt`进行测试，在不同的测试平台上，使用不同的例程和不同的模型进行测试，测试数据集：`datasets/coco/val2017`，阈值设置：person_thresh=0.5，conf_thresh=0.01，nms_thresh=0.6，flip=true，使用目标检测模型yolov5s_v6.1_3output_int8_4b.bmodel，性能测试结果如下：

|    测试平台   | 测试程序       | 测试模型                                      | decode_time | hrnet_preprocess_time | hrnet_inference_time | hrnet_postprocess_time |
| ----------- |---------------|------------------------------------------------|-------------|-----------------------|----------------------|------------------------|
|   SE7-32    | hrnet_pose.py |  hrnet_w32_256x192_fp32.bmodel                 | 8.39        | 9.21                  | 39.85                | 2.80                  |
|   SE7-32    | hrnet_pose.py |  hrnet_w32_256x192_fp16.bmodel                 | 8.35        | 9.19                  | 10.13                | 2.77                  |               
|   SE7-32    | hrnet_pose.py |  hrnet_w32_256x192_int8.bmodel                 | 8.31        | 9.13                  | 8.81                 | 2.77                  |
|   SE7-32    | hrnet_pose_bmcv.soc |  hrnet_w32_256x192_fp32.bmodel          | 4.44        | 2.19                  | 16.73                | 1.40                  |
|   SE7-32    | hrnet_pose_bmcv.soc |  hrnet_w32_256x192_fp16.bmodel          | 4.49        | 2.21                  | 1.89                 | 1.40                  |
|   SE7-32    | hrnet_pose_bmcv.soc |  hrnet_w32_256x192_int8.bmodel          | 4.48        | 2.21                  | 1.26                 | 1.41                  |
|   SE9-16    | hrnet_pose.py |  hrnet_w32_256x192_fp32.bmodel                 | 12.96        | 12.55                 | 161.07              | 3.92                  |
|   SE9-16    | hrnet_pose.py |  hrnet_w32_256x192_fp16.bmodel                 | 12.80        | 12.51                 | 27.95               | 3.91                  |
|   SE9-16    | hrnet_pose.py |  hrnet_w32_256x192_int8.bmodel                 | 13.02        | 12.53                 | 14.74               | 3.94                  |
|   SE9-16    | hrnet_pose_bmcv.soc |  hrnet_w32_256x192_fp32.bmodel           | 8.71         | 3.30                  | 76.28               | 1.55                  |
|   SE9-16    | hrnet_pose_bmcv.soc |  hrnet_w32_256x192_fp16.bmodel           | 8.56         | 3.29                  | 9.77                | 1.54                  |
|   SE9-16    | hrnet_pose_bmcv.soc |  hrnet_w32_256x192_int8.bmodel           | 10.14        | 3.28                  | 3.26                | 1.52                  |
|   SE9-16    | hrnet_pose.py |  hrnet_w32_256x192_fp32_2core.bmodel           | 13.00        | 12.64                 | 126.32              | 3.93                  |
|   SE9-16    | hrnet_pose.py |  hrnet_w32_256x192_fp16_2core.bmodel           | 12.76       | 12.51                  | 27.43               | 3.95                  |
|   SE9-16    | hrnet_pose.py |  hrnet_w32_256x192_int8_2core.bmodel           | 12.80       | 12.47                  | 14.43               | 3.91                  |
|   SE9-16    | hrnet_pose_bmcv.soc |  hrnet_w32_256x192_fp32_2core.bmodel     | 9.50        | 3.31                   | 58.85               | 1.54                  |
|   SE9-16    | hrnet_pose_bmcv.soc |  hrnet_w32_256x192_fp16_2core.bmodel     | 8.29        | 3.30                   | 9.51                | 1.54                  |
|   SE9-16    | hrnet_pose_bmcv.soc |  hrnet_w32_256x192_int8_2core.bmodel     | 8.44        | 3.29                   | 3.14                | 1.53                  |
|   SE9-8     | hrnet_pose.py |  hrnet_w32_256x192_fp32.bmodel                 | 12.90       | 12.41                | 160.91               | 3.81                   |
|   SE9-8     | hrnet_pose.py |  hrnet_w32_256x192_fp16.bmodel                 | 12.69       | 12.46                | 27.88                | 3.77                   |
|   SE9-8     | hrnet_pose.py |  hrnet_w32_256x192_int8.bmodel                 | 12.86       | 12.41                | 14.75                | 3.78                   |
|   SE9-8     | hrnet_pose_bmcv.soc |  hrnet_w32_256x192_fp32.bmodel           | 8.83        | 3.33                 | 76.26                | 1.73                   |
|   SE9-8     | hrnet_pose_bmcv.soc |  hrnet_w32_256x192_fp16.bmodel           | 9.05        | 3.32                 | 9.75                 | 1.71                   |
|   SE9-8     | hrnet_pose_bmcv.soc |  hrnet_w32_256x192_int8.bmodel           | 9.15        | 3.31                 | 3.24                 | 1.71                   |


> **测试说明**：  
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异，不同的阈值对后处理时间影响较大。
> 4. SE7-32的主控处理器为8核CA53@2.3GHz，SE9-16为8核CA53@1.6GHz，SE9-8为6核CA53@1.6GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 5. flip=true会提高mAP精度，但会增加前处理，推理和后处理的时间。