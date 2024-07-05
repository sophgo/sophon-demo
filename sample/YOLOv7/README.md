# YOLOv7

## 目录

- [YOLOv7](#yolov7)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
  - [3. 准备数据与模型](#3-准备数据与模型)
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

`YOLOv7`是基于anchor的one-stage目标检测算法,在准确率和速度上超越了以往的YOLO系列；
​本例程对[yolov7官方开源仓库](https://github.com/WongKinYiu/yolov7/tree/v0.1?tab=readme-ov-file)v0.1版本的模型[yolov7.pt](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt)和算法进行移植，使之能在SOPHON BM1684和BM1684X上进行推理测试。

## 2. 特性

* 支持BM1688/CV186X(SoC)、BM1684X(x86 PCIe、SoC)和BM1684(x86 PCIe、SoC、arm PCIe)
* 支持FP32、FP16(BM1684X/BM1688/CV186X)、INT8模型编译和推理
* 支持基于BMCV预处理的C++推理
* 支持基于OpenCV和BMCV预处理的Python推理
* 支持单batch和多batch模型推理
* 支持1个输出和3个输出模型推理
* 支持图片和视频测试

## 3. 准备数据与模型

Pytorch模型在编译前要导出成onnx模型，具体可参考[YOLOv7模型导出](./docs/YOLOv7_Export_Guide.md)。

​	同时，您需要准备用于测试的数据，如果量化模型，还要准备用于量化的数据集。

​	本例程在`scripts`目录下提供了相关模型和数据集的下载脚本`download.sh`，您也可以自己准备模型和数据集，并参考[4. 模型编译](#4-模型编译)进行模型转换。

```bash
# 安装7z和zip，若已安装请跳过，非ubuntu系统视情况使用yum或其他方式安装
sudo apt install unzip
sudo apt install p7zip; sudo apt install p7zip-full
chmod -R +x scripts/
./scripts/download.sh
```
下载的模型包括：
```
./models
├── BM1684
│   ├── yolov7_v0.1_3output_fp32_1b.bmodel  # 使用TPU-MLIR编译，用于BM1684的FP32 BModel，batch_size=1
│   ├── yolov7_v0.1_3output_fp32_4b.bmodel  # 使用TPU-MLIR编译，用于BM1684的FP32 BModel，batch_size=4
│   ├── yolov7_v0.1_3output_int8_1b.bmodel  # 使用TPU-MLIR编译，用于BM1684的INT8 BModel，batch_size=1
│   └── yolov7_v0.1_3output_int8_4b.bmodel  # 使用TPU-MLIR编译，用于BM1684的INT8 BModel，batch_size=4
├── BM1684X
│   ├── yolov7_v0.1_3output_fp16_1b.bmodel  # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=1
│   ├── yolov7_v0.1_3output_fp16_4b.bmodel  # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=4
│   ├── yolov7_v0.1_3output_fp32_1b.bmodel  # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=1
│   ├── yolov7_v0.1_3output_fp32_4b.bmodel  # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=4
│   ├── yolov7_v0.1_3output_int8_1b.bmodel  # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=1
│   └── yolov7_v0.1_3output_int8_4b.bmodel  # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=4
├── BM1688
│   ├── yolov7_v0.1_3output_fp16_1b_2core.bmodel  # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1，num_core=2
│   ├── yolov7_v0.1_3output_fp16_1b.bmodel        # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1
│   ├── yolov7_v0.1_3output_fp32_1b_2core.bmodel  # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1，num_core=2
│   ├── yolov7_v0.1_3output_fp32_1b.bmodel        # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1
│   ├── yolov7_v0.1_3output_int8_1b_2core.bmodel  # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=1，num_core=2
│   ├── yolov7_v0.1_3output_int8_1b.bmodel        # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=1
│   ├── yolov7_v0.1_3output_int8_4b_2core.bmodel  # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=4，num_core=2
│   └── yolov7_v0.1_3output_int8_4b.bmodel        # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=4
├── CV186X
│   ├── yolov7_v0.1_3output_fp16_1b.bmodel       # 使用TPU-MLIR编译，用于CV186X的FP16 BModel，batch_size=1
│   ├── yolov7_v0.1_3output_fp32_1b.bmodel       # 使用TPU-MLIR编译，用于CV186X的FP32 BModel，batch_size=1
│   ├── yolov7_v0.1_3output_int8_1b.bmodel       # 使用TPU-MLIR编译，用于CV186X的INT8 BModel，batch_size=1
│   └── yolov7_v0.1_3output_int8_4b.bmodel       # 使用TPU-MLIR编译，用于CV186X的INT8 BModel，batch_size=4
├── onnx
│   ├── yolov7_qtable
│   ├── yolov7_v0.1_3output_1b.onnx # 导出的onnx模型，batch_size=1
│   └── yolov7_v0.1_3output_4b.onnx # 导出的onnx模型，batch_size=4
└── torch
    └── yolov7_v0.1_3outputs.torchscript.pt # trace后的torchscript模型
```
下载的数据包括：
```
./datasets
├── test                                      # 测试图片
├── test_car_person_1080P.mp4                 # 测试视频
├── coco.names                                # coco类别名文件
├── coco128                                   # coco128数据集，用于模型量化
└── coco                                      
    ├── val2017_1000                               # coco val2017_1000数据集：coco val2017中随机抽取的1000张样本
    └── instances_val2017_1000.json                # coco val2017数据集标签文件，用于计算精度评价指标  
```


​	模型信息：

| 模型名称 | [yolov7.pt](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt)                                         |
| :------- | :------------------------------------------------------------------------------------------------------------------------- |
| 训练集   | MS COCO                                                                                                                    |
| 概述     | 80类通用目标检测                                                                                                           |
| 输入数据 | images, [batch_size, 3, 640, 640], FP32，NCHW，RGB planar                                                                  |
| 输出数据 | [batch_size, 3, 80, 80, 85], FP32 <br />[batch_size, 3, 40, 40, 85], FP32  <br />[batch_size, 3, 20, 20, 85], FP32  <br /> |
| 其他信息 | YOLO_ANCHORS: [12,16, 19,36, 40,28,  36,75, 76,55, 72,146,  142,110, 192,243, 459,401]                                     |
| 前处理   | BGR->RGB、/255.0                                                                                                           |
| 后处理   | nms等                                                                                                                      |



## 4. 模型编译

导出的模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。

模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#2-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的SDK中获取)。

- 生成FP32 BModel

本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684/BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684x #bm1684/bm1688/cv186x
```

执行上述命令会在`models/BM1684X/`下生成`yolov7_v0.1_3output_fp32_1b.bmodel`文件，即转换好的FP32 BModel。

- 生成FP16 BModel

本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x #bm1688/cv186x
```

执行上述命令会在`models/BM1684X/`下生成`yolov7_v0.1_3output_fp16_1b.bmodel`文件，即转换好的FP16 BModel。

- 生成INT8 BModel

本例程在`scripts`目录下提供了量化INT8 BModel的脚本，请注意修改`gen_int8bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台（**支持BM1684/BM1684X/BM1688/CV186X**），如：

```shell
./scripts/gen_int8bmodel_mlir.sh bm1684x #bm1684/bm1688/cv186x
```

上述脚本会在`models/BM1684X`下生成`yolov7_v0.1_3output_int8_1b.bmodel`等文件，即转换好的INT8 BModel。

## 5. 例程测试

- [C++例程](./cpp/README.md)
- [Python例程](./python/README.md)

## 6. 精度测试

### 6.1 测试方法

首先，参考[C++例程](cpp/README.md#32-测试图片)或[Python例程](python/README.md#22-测试图片)推理要测试的数据集，生成预测的json文件，注意修改数据集(datasets/coco/val2017_1000)和相关参数(conf_thresh=0.001、nms_thresh=0.65)。  
然后，使用`tools`目录下的`eval_coco.py`脚本，将测试生成的json文件与测试集标签json文件进行对比，计算出目标检测的评价指标，命令如下：

```bash
# 安装pycocotools，若已安装请跳过
pip3 install pycocotools
# 请根据实际情况修改程序路径和json文件路径
python3 tools/eval_coco.py --gt_path datasets/coco/instances_val2017_1000.json --result_json results/yolov7_v0.1_3output_fp32_1b.bmodel_val2017_1000_opencv_python_result.json
```
### 6.2 测试结果
在coco2017val_1000数据集上，精度测试结果如下：

| 测试平台 | 测试程序         | 测试模型                                 | AP@IoU=0.5:0.95 | AP@IoU=0.5 |
| -------- | ---------------- | ---------------------------------------- | --------------- | ---------- |
| SE5-16   | yolov7_opencv.py | yolov7_v0.1_3output_fp32_1b.bmodel       | 0.514           | 0.699      |
| SE5-16   | yolov7_opencv.py | yolov7_v0.1_3output_int8_1b.bmodel       | 0.505           | 0.696      |
| SE5-16   | yolov7_opencv.py | yolov7_v0.1_3output_int8_4b.bmodel       | 0.505           | 0.696      |
| SE5-16   | yolov7_bmcv.py   | yolov7_v0.1_3output_fp32_1b.bmodel       | 0.504           | 0.687      |
| SE5-16   | yolov7_bmcv.py   | yolov7_v0.1_3output_int8_1b.bmodel       | 0.497           | 0.684      |
| SE5-16   | yolov7_bmcv.py   | yolov7_v0.1_3output_int8_4b.bmodel       | 0.497           | 0.684      |
| SE5-16   | yolov7_bmcv.soc  | yolov7_v0.1_3output_fp32_1b.bmodel       | 0.494           | 0.696      |
| SE5-16   | yolov7_bmcv.soc  | yolov7_v0.1_3output_int8_1b.bmodel       | 0.487           | 0.691      |
| SE5-16   | yolov7_bmcv.soc  | yolov7_v0.1_3output_int8_4b.bmodel       | 0.487           | 0.691      |
| SE7-32   | yolov7_opencv.py | yolov7_v0.1_3output_fp32_1b.bmodel       | 0.514           | 0.699      |
| SE7-32   | yolov7_opencv.py | yolov7_v0.1_3output_fp16_1b.bmodel       | 0.514           | 0.700      |
| SE7-32   | yolov7_opencv.py | yolov7_v0.1_3output_int8_1b.bmodel       | 0.511           | 0.698      |
| SE7-32   | yolov7_opencv.py | yolov7_v0.1_3output_int8_4b.bmodel       | 0.511           | 0.698      |
| SE7-32   | yolov7_bmcv.py   | yolov7_v0.1_3output_fp32_1b.bmodel       | 0.504           | 0.687      |
| SE7-32   | yolov7_bmcv.py   | yolov7_v0.1_3output_fp16_1b.bmodel       | 0.504           | 0.688      |
| SE7-32   | yolov7_bmcv.py   | yolov7_v0.1_3output_int8_1b.bmodel       | 0.501           | 0.687      |
| SE7-32   | yolov7_bmcv.py   | yolov7_v0.1_3output_int8_4b.bmodel       | 0.501           | 0.687      |
| SE7-32   | yolov7_bmcv.soc  | yolov7_v0.1_3output_fp32_1b.bmodel       | 0.493           | 0.696      |
| SE7-32   | yolov7_bmcv.soc  | yolov7_v0.1_3output_fp16_1b.bmodel       | 0.494           | 0.696      |
| SE7-32   | yolov7_bmcv.soc  | yolov7_v0.1_3output_int8_1b.bmodel       | 0.492           | 0.698      |
| SE7-32   | yolov7_bmcv.soc  | yolov7_v0.1_3output_int8_4b.bmodel       | 0.492           | 0.698      |
| SE9-16   | yolov7_opencv.py | yolov7_v0.1_3output_fp32_1b.bmodel       | 0.514           | 0.699      |
| SE9-16   | yolov7_opencv.py | yolov7_v0.1_3output_fp16_1b.bmodel       | 0.514           | 0.699      |
| SE9-16   | yolov7_opencv.py | yolov7_v0.1_3output_int8_1b.bmodel       | 0.510           | 0.699      |
| SE9-16   | yolov7_opencv.py | yolov7_v0.1_3output_int8_4b.bmodel       | 0.510           | 0.699      |
| SE9-16   | yolov7_bmcv.py   | yolov7_v0.1_3output_fp32_1b.bmodel       | 0.504           | 0.687      |
| SE9-16   | yolov7_bmcv.py   | yolov7_v0.1_3output_fp16_1b.bmodel       | 0.503           | 0.687      |
| SE9-16   | yolov7_bmcv.py   | yolov7_v0.1_3output_int8_1b.bmodel       | 0.500           | 0.686      |
| SE9-16   | yolov7_bmcv.py   | yolov7_v0.1_3output_int8_4b.bmodel       | 0.500           | 0.686      |
| SE9-16   | yolov7_bmcv.soc  | yolov7_v0.1_3output_fp32_1b.bmodel       | 0.493           | 0.696      |
| SE9-16   | yolov7_bmcv.soc  | yolov7_v0.1_3output_fp16_1b.bmodel       | 0.493           | 0.696      |
| SE9-16   | yolov7_bmcv.soc  | yolov7_v0.1_3output_int8_1b.bmodel       | 0.490           | 0.695      |
| SE9-16   | yolov7_bmcv.soc  | yolov7_v0.1_3output_int8_4b.bmodel       | 0.490           | 0.695      |
| SE9-16   | yolov7_opencv.py | yolov7_v0.1_3output_fp32_1b_2core.bmodel | 0.514           | 0.699      |
| SE9-16   | yolov7_opencv.py | yolov7_v0.1_3output_fp16_1b_2core.bmodel | 0.514           | 0.699      |
| SE9-16   | yolov7_opencv.py | yolov7_v0.1_3output_int8_1b_2core.bmodel | 0.473           | 0.695      |
| SE9-16   | yolov7_opencv.py | yolov7_v0.1_3output_int8_4b_2core.bmodel | 0.473           | 0.695      |
| SE9-16   | yolov7_bmcv.py   | yolov7_v0.1_3output_fp32_1b_2core.bmodel | 0.504           | 0.687      |
| SE9-16   | yolov7_bmcv.py   | yolov7_v0.1_3output_fp16_1b_2core.bmodel | 0.504           | 0.687      |
| SE9-16   | yolov7_bmcv.py   | yolov7_v0.1_3output_int8_1b_2core.bmodel | 0.463           | 0.681      |
| SE9-16   | yolov7_bmcv.py   | yolov7_v0.1_3output_int8_4b_2core.bmodel | 0.463           | 0.681      |
| SE9-16   | yolov7_bmcv.soc  | yolov7_v0.1_3output_fp32_1b_2core.bmodel | 0.493           | 0.696      |
| SE9-16   | yolov7_bmcv.soc  | yolov7_v0.1_3output_fp16_1b_2core.bmodel | 0.494           | 0.696      |
| SE9-16   | yolov7_bmcv.soc  | yolov7_v0.1_3output_int8_1b_2core.bmodel | 0.457           | 0.688      |
| SE9-16   | yolov7_bmcv.soc  | yolov7_v0.1_3output_int8_4b_2core.bmodel | 0.457           | 0.688      |
| SE9-8    | yolov7_opencv.py | yolov7_v0.1_3output_fp32_1b.bmodel       | 0.514           | 0.699      |
| SE9-8    | yolov7_opencv.py | yolov7_v0.1_3output_fp16_1b.bmodel       | 0.514           | 0.699      |
| SE9-8    | yolov7_opencv.py | yolov7_v0.1_3output_int8_1b.bmodel       | 0.510           | 0.699      |
| SE9-8    | yolov7_opencv.py | yolov7_v0.1_3output_int8_4b.bmodel       | 0.510           | 0.699      |
| SE9-8    | yolov7_bmcv.py   | yolov7_v0.1_3output_fp32_1b.bmodel       | 0.504           | 0.687      |
| SE9-8    | yolov7_bmcv.py   | yolov7_v0.1_3output_fp16_1b.bmodel       | 0.503           | 0.687      |
| SE9-8    | yolov7_bmcv.py   | yolov7_v0.1_3output_int8_1b.bmodel       | 0.500           | 0.686      |
| SE9-8    | yolov7_bmcv.py   | yolov7_v0.1_3output_int8_4b.bmodel       | 0.500           | 0.686      |
| SE9-8    | yolov7_bmcv.soc  | yolov7_v0.1_3output_fp32_1b.bmodel       | 0.493           | 0.696      |
| SE9-8    | yolov7_bmcv.soc  | yolov7_v0.1_3output_fp16_1b.bmodel       | 0.493           | 0.696      |
| SE9-8    | yolov7_bmcv.soc  | yolov7_v0.1_3output_int8_1b.bmodel       | 0.490           | 0.695      |
| SE9-8    | yolov7_bmcv.soc  | yolov7_v0.1_3output_int8_4b.bmodel       | 0.490           | 0.695      |

> **测试说明**：  
1. 由于sdk版本之间可能存在差异，实际运行结果与本表有<0.01的精度误差是正常的；
2. AP@IoU=0.5:0.95为area=all对应的指标；
3. 在搭载了相同TPU和SOPHONSDK的PCIe或SoC平台上，相同程序的精度一致，SE5系列对应BM1684，SE7系列对应BM1684X，SE9系列中，SE9-16对应BM1688，SE9-8对应CV186X。

## 7. 性能测试
### 7.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684/yolov7_v0.1_3output_fp32_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

| 测试模型                                        | calculate time(ms) |
| ----------------------------------------------- | ------------------ |
| BM1684/yolov7_v0.1_3output_fp32_1b.bmodel       | 87.2               |
| BM1684/yolov7_v0.1_3output_int8_1b.bmodel       | 48.9               |
| BM1684/yolov7_v0.1_3output_int8_4b.bmodel       | 19.4               |
| BM1684X/yolov7_v0.1_3output_fp32_1b.bmodel      | 97.8               |
| BM1684X/yolov7_v0.1_3output_fp16_1b.bmodel      | 19.3               |
| BM1684X/yolov7_v0.1_3output_fp16_4b.bmodel      | 18.4               |
| BM1684X/yolov7_v0.1_3output_int8_1b.bmodel      | 9.1                |
| BM1684X/yolov7_v0.1_3output_int8_4b.bmodel      | 8.4                |
| BM1688/yolov7_v0.1_3output_fp32_1b.bmodel       | 582.54             |
| BM1688/yolov7_v0.1_3output_fp16_1b.bmodel       | 129.02             |
| BM1688/yolov7_v0.1_3output_int8_1b.bmodel       | 33.42              |
| BM1688/yolov7_v0.1_3output_int8_4b.bmodel       | 32.90              |
| BM1688/yolov7_v0.1_3output_fp32_1b_2core.bmodel | 318.84             |
| BM1688/yolov7_v0.1_3output_fp16_1b_2core.bmodel | 87.91              |
| BM1688/yolov7_v0.1_3output_int8_1b_2core.bmodel | 19.66              |
| BM1688/yolov7_v0.1_3output_int8_4b_2core.bmodel | 16.78              |
| CV186X/yolov7_v0.1_3output_fp32_1b.bmodel       | 577.94             |
| CV186X/yolov7_v0.1_3output_fp16_1b.bmodel       | 123.22             |
| CV186X/yolov7_v0.1_3output_int8_1b.bmodel       | 33.50              |
| CV186X/yolov7_v0.1_3output_int8_4b.bmodel       | 33.01              |


> **测试说明**：  
1. 性能测试结果具有一定的波动性；
2. `calculate time`已折算为平均每张图片的推理时间；
3. SoC和PCIe的测试结果基本一致。

### 7.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++和Python例程打印的时间已经折算为单张图片的处理时间。

在不同的测试平台上，使用不同的例程、模型测试`datasets/coco/val2017_1000`，conf_thresh=0.5，nms_thresh=0.5，性能测试结果如下：

| 测试平台 | 测试程序         | 测试模型                                 | decode_time | preprocess_time | inference_time | postprocess_time |
| -------- | ---------------- | ---------------------------------------- | ----------- | --------------- | -------------- | ---------------- |
| SE5-16   | yolov7_opencv.py | yolov7_v0.1_3output_fp32_1b.bmodel       | 20.10       | 26.29           | 93.05          | 111.12           |
| SE5-16   | yolov7_opencv.py | yolov7_v0.1_3output_int8_1b.bmodel       | 13.99       | 26.35           | 71.77          | 109.78           |
| SE5-16   | yolov7_opencv.py | yolov7_v0.1_3output_int8_4b.bmodel       | 13.85       | 23.91           | 40.65          | 112.05           |
| SE5-16   | yolov7_bmcv.py   | yolov7_v0.1_3output_fp32_1b.bmodel       | 3.57        | 2.82            | 89.29          | 106.19           |
| SE5-16   | yolov7_bmcv.py   | yolov7_v0.1_3output_int8_1b.bmodel       | 3.57        | 2.30            | 54.98          | 105.83           |
| SE5-16   | yolov7_bmcv.py   | yolov7_v0.1_3output_int8_4b.bmodel       | 3.45        | 2.12            | 24.84          | 109.49           |
| SE5-16   | yolov7_bmcv.soc  | yolov7_v0.1_3output_fp32_1b.bmodel       | 4.87        | 1.55            | 82.58          | 18.60            |
| SE5-16   | yolov7_bmcv.soc  | yolov7_v0.1_3output_int8_1b.bmodel       | 4.86        | 1.54            | 48.19          | 18.57            |
| SE5-16   | yolov7_bmcv.soc  | yolov7_v0.1_3output_int8_4b.bmodel       | 4.72        | 1.47            | 19.05          | 18.48            |
| SE7-32   | yolov7_opencv.py | yolov7_v0.1_3output_fp32_1b.bmodel       | 18.38       | 27.02           | 111.60         | 109.77           |
| SE7-32   | yolov7_opencv.py | yolov7_v0.1_3output_fp16_1b.bmodel       | 14.11       | 27.70           | 35.96          | 109.32           |
| SE7-32   | yolov7_opencv.py | yolov7_v0.1_3output_int8_1b.bmodel       | 14.01       | 27.78           | 20.83          | 109.39           |
| SE7-32   | yolov7_opencv.py | yolov7_v0.1_3output_int8_4b.bmodel       | 13.96       | 25.33           | 18.73          | 112.40           |
| SE7-32   | yolov7_bmcv.py   | yolov7_v0.1_3output_fp32_1b.bmodel       | 3.04        | 2.34            | 106.45         | 104.30           |
| SE7-32   | yolov7_bmcv.py   | yolov7_v0.1_3output_fp16_1b.bmodel       | 3.02        | 2.35            | 30.94          | 103.90           |
| SE7-32   | yolov7_bmcv.py   | yolov7_v0.1_3output_int8_1b.bmodel       | 3.03        | 2.34            | 15.72          | 104.00           |
| SE7-32   | yolov7_bmcv.py   | yolov7_v0.1_3output_int8_4b.bmodel       | 2.90        | 2.17            | 14.37          | 108.36           |
| SE7-32   | yolov7_bmcv.soc  | yolov7_v0.1_3output_fp32_1b.bmodel       | 4.31        | 0.74            | 99.85          | 18.65            |
| SE7-32   | yolov7_bmcv.soc  | yolov7_v0.1_3output_fp16_1b.bmodel       | 4.29        | 0.74            | 24.34          | 18.65            |
| SE7-32   | yolov7_bmcv.soc  | yolov7_v0.1_3output_int8_1b.bmodel       | 4.31        | 0.74            | 9.11           | 18.66            |
| SE7-32   | yolov7_bmcv.soc  | yolov7_v0.1_3output_int8_4b.bmodel       | 4.16        | 0.71            | 8.70           | 18.54            |
| SE9-16   | yolov7_opencv.py | yolov7_v0.1_3output_fp32_1b.bmodel       | 22.21       | 37.03           | 581.77         | 150.65           |
| SE9-16   | yolov7_opencv.py | yolov7_v0.1_3output_fp16_1b.bmodel       | 19.42       | 36.64           | 128.73         | 150.99           |
| SE9-16   | yolov7_opencv.py | yolov7_v0.1_3output_int8_1b.bmodel       | 19.46       | 36.37           | 44.55          | 150.91           |
| SE9-16   | yolov7_opencv.py | yolov7_v0.1_3output_int8_4b.bmodel       | 19.26       | 33.41           | 41.89          | 150.92           |
| SE9-16   | yolov7_bmcv.py   | yolov7_v0.1_3output_fp32_1b.bmodel       | 4.44        | 5.04            | 577.06         | 143.57           |
| SE9-16   | yolov7_bmcv.py   | yolov7_v0.1_3output_fp16_1b.bmodel       | 4.37        | 5.06            | 123.28         | 143.58           |
| SE9-16   | yolov7_bmcv.py   | yolov7_v0.1_3output_int8_1b.bmodel       | 4.36        | 5.00            | 38.96          | 143.61           |
| SE9-16   | yolov7_bmcv.py   | yolov7_v0.1_3output_int8_4b.bmodel       | 4.27        | 4.75            | 37.47          | 150.91           |
| SE9-16   | yolov7_bmcv.soc  | yolov7_v0.1_3output_fp32_1b.bmodel       | 5.92        | 1.82            | 567.47         | 26.02            |
| SE9-16   | yolov7_bmcv.soc  | yolov7_v0.1_3output_fp16_1b.bmodel       | 5.88        | 1.83            | 113.91         | 25.95            |
| SE9-16   | yolov7_bmcv.soc  | yolov7_v0.1_3output_int8_1b.bmodel       | 5.84        | 1.82            | 29.66          | 25.92            |
| SE9-16   | yolov7_bmcv.soc  | yolov7_v0.1_3output_int8_4b.bmodel       | 5.68        | 1.74            | 29.50          | 25.83            |
| SE9-16   | yolov7_opencv.py | yolov7_v0.1_3output_fp32_1b_2core.bmodel | 19.49       | 36.79           | 318.75         | 151.24           |
| SE9-16   | yolov7_opencv.py | yolov7_v0.1_3output_fp16_1b_2core.bmodel | 19.28       | 36.04           | 87.75          | 151.02           |
| SE9-16   | yolov7_opencv.py | yolov7_v0.1_3output_int8_1b_2core.bmodel | 19.46       | 36.73           | 31.93          | 151.16           |
| SE9-16   | yolov7_opencv.py | yolov7_v0.1_3output_int8_4b_2core.bmodel | 19.28       | 33.22           | 26.07          | 150.90           |
| SE9-16   | yolov7_bmcv.py   | yolov7_v0.1_3output_fp32_1b_2core.bmodel | 4.43        | 5.02            | 313.23         | 143.94           |
| SE9-16   | yolov7_bmcv.py   | yolov7_v0.1_3output_fp16_1b_2core.bmodel | 4.37        | 5.03            | 82.11          | 143.78           |
| SE9-16   | yolov7_bmcv.py   | yolov7_v0.1_3output_int8_1b_2core.bmodel | 4.37        | 5.04            | 26.54          | 143.44           |
| SE9-16   | yolov7_bmcv.py   | yolov7_v0.1_3output_int8_4b_2core.bmodel | 4.40        | 4.70            | 21.64          | 150.61           |
| SE9-16   | yolov7_bmcv.soc  | yolov7_v0.1_3output_fp32_1b_2core.bmodel | 6.28        | 1.82            | 303.79         | 25.97            |
| SE9-16   | yolov7_bmcv.soc  | yolov7_v0.1_3output_fp16_1b_2core.bmodel | 6.94        | 1.82            | 72.65          | 25.99            |
| SE9-16   | yolov7_bmcv.soc  | yolov7_v0.1_3output_int8_1b_2core.bmodel | 5.85        | 1.83            | 17.07          | 26.01            |
| SE9-16   | yolov7_bmcv.soc  | yolov7_v0.1_3output_int8_4b_2core.bmodel | 5.67        | 1.74            | 13.64          | 25.79            |
| SE9-8    | yolov7_opencv.py | yolov7_v0.1_3output_fp32_1b.bmodel       | 33.68       | 35.82           | 592.63         | 153.60           |
| SE9-8    | yolov7_opencv.py | yolov7_v0.1_3output_fp16_1b.bmodel       | 25.47       | 36.44           | 137.69         | 150.43           |
| SE9-8    | yolov7_opencv.py | yolov7_v0.1_3output_int8_1b.bmodel       | 21.03       | 36.40           | 47.95          | 150.02           |
| SE9-8    | yolov7_opencv.py | yolov7_v0.1_3output_int8_4b.bmodel       | 20.88       | 32.59           | 45.06          | 149.44           |
| SE9-8    | yolov7_bmcv.py   | yolov7_v0.1_3output_fp32_1b.bmodel       | 4.22        | 4.86            | 587.39         | 143.62           |
| SE9-8    | yolov7_bmcv.py   | yolov7_v0.1_3output_fp16_1b.bmodel       | 4.21        | 4.90            | 132.43         | 143.64           |
| SE9-8    | yolov7_bmcv.py   | yolov7_v0.1_3output_int8_1b.bmodel       | 5.82        | 4.87            | 42.63          | 143.51           |
| SE9-8    | yolov7_bmcv.py   | yolov7_v0.1_3output_int8_4b.bmodel       | 4.09        | 4.57            | 41.92          | 149.68           |
| SE9-8    | yolov7_bmcv.soc  | yolov7_v0.1_3output_fp32_1b.bmodel       | 5.88        | 1.81            | 577.88         | 26.00            |
| SE9-8    | yolov7_bmcv.soc  | yolov7_v0.1_3output_fp16_1b.bmodel       | 6.74        | 1.81            | 123.12         | 25.96            |
| SE9-8    | yolov7_bmcv.soc  | yolov7_v0.1_3output_int8_1b.bmodel       | 5.63        | 1.81            | 33.37          | 25.96            |
| SE9-8    | yolov7_bmcv.soc  | yolov7_v0.1_3output_int8_4b.bmodel       | 5.47        | 1.72            | 32.98          | 25.80            |



> **测试说明**：  
1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
3. SE5-16/SE7-32的主控处理器均为8核CA53@2.3GHz，SE9-16为8核CA53@1.6GHz，SE9-8为6核CA53@1.6GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异。 

## 8. FAQ
请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。