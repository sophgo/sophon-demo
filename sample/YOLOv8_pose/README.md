# YOLOv8-pose

## 目录

* [1. 简介](#1-简介)
* [2. 特性](#2-特性)
* [3. 准备模型与数据](#3-准备模型与数据)
* [4. 模型编译](#4-模型编译)
* [5. 例程测试](#5-例程测试)
* [6. 精度测试](#6-精度测试)
  * [6.1 测试方法](#61-测试方法)
  * [6.2 测试结果](#62-测试结果)
* [7. 性能测试](#7-性能测试)
  * [7.1 bmrt_test](#71-bmrt_test)
  * [7.2 程序运行性能](#72-程序运行性能)
* [8. FAQ](#8-faq)
  
## 1. 简介
​YOLOv8是YOLO系列的的一个重大更新版本，它抛弃了以往的YOLO系类模型使用的Anchor-Base，采用了Anchor-Free的思想。YOLOv8建立在YOLO系列成功的基础上，通过对网络结构的改造，进一步提升其性能和灵活性。本例程对[​YOLOv8官方开源仓库](https://github.com/ultralytics/ultralytics)的模型和算法进行移植，使之能在SOPHON BM1684/BM1684X/BM1688/CV186X上进行推理测试。

## 2. 特性
* 支持BM1688/CV186X(SoC)、BM1684X(x86 PCIe、SoC)、BM1684(x86 PCIe、SoC、arm PCIe)
* 支持FP32、FP16(BM1684X/BM1688/CV186X)、INT8模型编译和推理
* 支持基于BMCV预处理的C++推理
* 支持基于OpenCV和BMCV预处理的Python推理
* 支持单batch和多batch模型推理
* 支持1个输出模型推理
* 支持图片和视频测试
 
## 3. 准备模型与数据
建议使用TPU-MLIR编译BModel，在使用TPU-MLIR编译前需要导出ONNX模型。具体可参考[YOLOv8模型导出](./docs/YOLOv8_Export_Guide.md)。

​同时，您需要准备用于测试的数据集，如果量化模型，还要准备用于量化的数据集。

​本例程在`scripts`目录下提供了相关模型和数据的下载脚本`download.sh`，您也可以自己准备模型和数据集，并参考[4. 模型编译](#4-模型编译)进行模型转换。

```bash
# 安装unzip，若已安装请跳过，非ubuntu系统视情况使用yum或其他方式安装
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```

执行后，模型保存至`models/`，测试数据集下载并解压至`datasets/test/`，精度测试数据集下载并解压至`datasets/coco/val2017_1000/`，量化数据集下载并解压至`datasets/coco128/`

```
下载的模型包括：
./models
├── BM1684
│   ├── yolov8s-pose_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684的FP32 BModel，batch_size=1
│   ├── yolov8s-pose_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684的INT8 BModel，batch_size=1
│   └── yolov8s-pose_int8_4b.bmodel   # 使用TPU-MLIR编译，用于BM1684的INT8 BModel，batch_size=4
├── BM1684X
│   ├── yolov8s-pose_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=1
│   ├── yolov8s-pose_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=1
│   ├── yolov8s-pose_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=1
│   └── yolov8s-pose_int8_4b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=4
├── BM1688
│   ├── yolov8s-pose_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1, num_core=1
│   ├── yolov8s-pose_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1, num_core=1
│   ├── yolov8s-pose_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=1, num_core=1
│   ├── yolov8s-pose_int8_4b.bmodel   # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=4, num_core=1
│   ├── yolov8s-pose_fp32_1b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1, num_core=2
│   ├── yolov8s-pose_fp16_1b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1, num_core=2
│   ├── yolov8s-pose_int8_1b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=1, num_core=2
│   └── yolov8s-pose_int8_4b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=4, num_core=2
├── CV186X
│   ├── yolov8s-pose_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于CV186X的FP32 BModel，batch_size=1
│   ├── yolov8s-pose_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于CV186X的FP16 BModel，batch_size=1
│   ├── yolov8s-pose_int8_1b.bmodel   # 使用TPU-MLIR编译，用于CV186X的INT8 BModel，batch_size=1
│   └── yolov8s-pose_int8_4b.bmodel   # 使用TPU-MLIR编译，用于CV186X的INT8 BModel，batch_size=4
│── torch
│   └── yolov8s-pose.pt        # pytorch模型
└── onnx
    └── yolov8s-pose.onnx      # 导出的动态onnx模型
```

下载的数据包括：
```
./datasets
├── test                                      # 测试图片
├── dance_1080P.mp4                           # 测试视频
├── coco128                                   # coco128数据集，用于模型量化
└── coco                                      
    ├── val2017_1000                          # coco val2017_1000数据集：coco val2017中随机抽取的1000张样本
    └── person_keypoints_val2017_1000.json    # coco val2017_1000数据集标签文件，用于计算精度评价指标  
```

## 4. 模型编译
导出的模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。建议使用TPU-MLIR编译BModel。

模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md##1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的SDK中获取)。

- 生成FP32 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684/BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684 #bm1684x/bm1688/cv186x
```

​执行上述命令会在`models/BM1684`等文件夹下生成`yolov8s-pose_fp32_1b.bmodel`文件，即转换好的FP32 BModel。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x #bm1688/cv186x
```

​执行上述命令会在`models/BM1684X/`等文件夹下生成`yolov8s-pose_fp16_1b.bmodel`文件，即转换好的FP16 BModel。

- 生成INT8 BModel

​本例程在`scripts`目录下提供了量化INT8 BModel的脚本，请注意修改`gen_int8bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台（**支持BM1684/BM1684X/BM1688**），如：

```shell
./scripts/gen_int8bmodel_mlir.sh bm1684 #bm1684x/bm1688/cv186x
```

​上述脚本会在`models/BM1684`等文件夹下生成`yolov8s-pose_int8_1b.bmodel`等文件，即转换好的INT8 BModel。


## 5. 例程测试
- [C++例程](./cpp/README.md)
- [Python例程](./python/README.md)

## 6. 精度测试
### 6.1 测试方法

首先，参考[C++例程](cpp/README.md#32-测试图片)或[Python例程](python/README.md#22-测试图片)推理要测试的数据集，生成预测的json文件，注意修改数据集(datasets/coco/val2017_1000)和相关参数(conf_thresh=0.001、nms_thresh=0.6)。  
然后，使用`tools`目录下的`eval_coco.py`脚本，将测试生成的json文件与测试集标签json文件进行对比，计算出目标检测的评价指标，命令如下：
```bash
# 安装pycocotools，若已安装请跳过
pip3 install pycocotools
# 请根据实际情况修改程序路径和json文件路径
python3 tools/eval_coco.py --gt_path datasets/coco/person_keypoints_val2017_1000.json --result_json results/yolov8s-pose_fp32_1b.bmodel_val2017_1000_opencv_python_result.json
```

### 6.2 测试结果
在`datasets/coco/val2017_1000`数据集上，参数设置为`nms_thresh=0.7,conf_thresh=0.001`,精度测试结果如下：
|   测试平台    |      测试程序     |               测试模型               | AP@IoU=0.5:0.95 | AP@IoU=0.5 |
| ------------ | ---------------- | ---------------------------------- | --------------- | ---------- |
| SE5-16       | yolov8_opencv.py | yolov8s-pose_fp32_1b.bmodel        |      0.575      |    0.801   |
| SE5-16       | yolov8_opencv.py | yolov8s-pose_int8_1b.bmodel        |      0.491      |    0.769   |
| SE5-16       | yolov8_opencv.py | yolov8s-pose_int8_4b.bmodel        |      0.491      |    0.769   |
| SE5-16       | yolov8_bmcv.py   | yolov8s-pose_fp32_1b.bmodel        |      0.576      |    0.802   |
| SE5-16       | yolov8_bmcv.py   | yolov8s-pose_int8_1b.bmodel        |      0.490      |    0.762   |
| SE5-16       | yolov8_bmcv.py   | yolov8s-pose_int8_4b.bmodel        |      0.490      |    0.762   |
| SE5-16       | yolov8_bmcv.soc  | yolov8s-pose_fp32_1b.bmodel        |      0.576      |    0.802   |
| SE5-16       | yolov8_bmcv.soc  | yolov8s-pose_int8_1b.bmodel        |      0.491      |    0.771   |
| SE5-16       | yolov8_bmcv.soc  | yolov8s-pose_int8_4b.bmodel        |      0.491      |    0.771   |
| SE7-32       | yolov8_opencv.py | yolov8s-pose_fp32_1b.bmodel        |      0.575      |    0.801   |
| SE7-32       | yolov8_opencv.py | yolov8s-pose_fp16_1b.bmodel        |      0.574      |    0.801   |
| SE7-32       | yolov8_opencv.py | yolov8s-pose_int8_1b.bmodel        |      0.553      |    0.801   |
| SE7-32       | yolov8_opencv.py | yolov8s-pose_int8_4b.bmodel        |      0.553      |    0.801   |
| SE7-32       | yolov8_bmcv.py   | yolov8s-pose_fp32_1b.bmodel        |      0.574      |    0.793   |
| SE7-32       | yolov8_bmcv.py   | yolov8s-pose_fp16_1b.bmodel        |      0.574      |    0.793   |
| SE7-32       | yolov8_bmcv.py   | yolov8s-pose_int8_1b.bmodel        |      0.554      |    0.801   |
| SE7-32       | yolov8_bmcv.py   | yolov8s-pose_int8_4b.bmodel        |      0.554      |    0.801   |
| SE7-32       | yolov8_bmcv.soc  | yolov8s-pose_fp32_1b.bmodel        |      0.574      |    0.802   |
| SE7-32       | yolov8_bmcv.soc  | yolov8s-pose_fp16_1b.bmodel        |      0.575      |    0.802   |
| SE7-32       | yolov8_bmcv.soc  | yolov8s-pose_int8_1b.bmodel        |      0.553      |    0.791   |
| SE7-32       | yolov8_bmcv.soc  | yolov8s-pose_int8_4b.bmodel        |      0.553      |    0.791   |
| SE9-16       | yolov8_opencv.py | yolov8s-pose_fp32_1b.bmodel        |      0.575      |    0.801   |
| SE9-16       | yolov8_opencv.py | yolov8s-pose_fp16_1b.bmodel        |      0.574      |    0.801   |
| SE9-16       | yolov8_opencv.py | yolov8s-pose_int8_1b.bmodel        |      0.553      |    0.801   |
| SE9-16       | yolov8_opencv.py | yolov8s-pose_int8_4b.bmodel        |      0.553      |    0.801   |
| SE9-16       | yolov8_bmcv.py   | yolov8s-pose_fp32_1b.bmodel        |      0.574      |    0.793   |
| SE9-16       | yolov8_bmcv.py   | yolov8s-pose_fp16_1b.bmodel        |      0.574      |    0.793   |
| SE9-16       | yolov8_bmcv.py   | yolov8s-pose_int8_1b.bmodel        |      0.550      |    0.791   |
| SE9-16       | yolov8_bmcv.py   | yolov8s-pose_int8_4b.bmodel        |      0.550      |    0.791   |
| SE9-16       | yolov8_bmcv.soc  | yolov8s-pose_fp32_1b.bmodel        |      0.574      |    0.793   |
| SE9-16       | yolov8_bmcv.soc  | yolov8s-pose_fp16_1b.bmodel        |      0.574      |    0.793   |
| SE9-16       | yolov8_bmcv.soc  | yolov8s-pose_int8_1b.bmodel        |      0.551      |    0.800   |
| SE9-16       | yolov8_bmcv.soc  | yolov8s-pose_int8_4b.bmodel        |      0.551      |    0.800   |
| SE9-16       | yolov8_opencv.py | yolov8s-pose_fp32_1b_2core.bmodel  |      0.575      |    0.801   |
| SE9-16       | yolov8_opencv.py | yolov8s-pose_fp16_1b_2core.bmodel  |      0.574      |    0.801   |
| SE9-16       | yolov8_opencv.py | yolov8s-pose_int8_1b_2core.bmodel  |      0.553      |    0.801   |
| SE9-16       | yolov8_opencv.py | yolov8s-pose_int8_4b_2core.bmodel  |      0.553      |    0.801   |
| SE9-16       | yolov8_bmcv.py   | yolov8s-pose_fp32_1b_2core.bmodel  |      0.574      |    0.793   |
| SE9-16       | yolov8_bmcv.py   | yolov8s-pose_fp16_1b_2core.bmodel  |      0.574      |    0.793   |
| SE9-16       | yolov8_bmcv.py   | yolov8s-pose_int8_1b_2core.bmodel  |      0.550      |    0.791   |
| SE9-16       | yolov8_bmcv.py   | yolov8s-pose_int8_4b_2core.bmodel  |      0.550      |    0.791   |
| SE9-16       | yolov8_bmcv.soc  | yolov8s-pose_fp32_1b_2core.bmodel  |      0.574      |    0.793   |
| SE9-16       | yolov8_bmcv.soc  | yolov8s-pose_fp16_1b_2core.bmodel  |      0.574      |    0.793   |
| SE9-16       | yolov8_bmcv.soc  | yolov8s-pose_int8_1b_2core.bmodel  |      0.551      |    0.800   |
| SE9-16       | yolov8_bmcv.soc  | yolov8s-pose_int8_4b_2core.bmodel  |      0.551      |    0.800   |
| SE9-8        | yolov8_opencv.py | yolov8s-pose_fp32_1b.bmodel        |      0.575      |    0.801   |
| SE9-8        | yolov8_opencv.py | yolov8s-pose_fp16_1b.bmodel        |      0.574      |    0.801   |
| SE9-8        | yolov8_opencv.py | yolov8s-pose_int8_1b.bmodel        |      0.553      |    0.801   |
| SE9-8        | yolov8_opencv.py | yolov8s-pose_int8_4b.bmodel        |      0.553      |    0.801   |
| SE9-8        | yolov8_bmcv.py   | yolov8s-pose_fp32_1b.bmodel        |      0.574      |    0.793   |
| SE9-8        | yolov8_bmcv.py   | yolov8s-pose_fp16_1b.bmodel        |      0.574      |    0.793   |
| SE9-8        | yolov8_bmcv.py   | yolov8s-pose_int8_1b.bmodel        |      0.550      |    0.791   |
| SE9-8        | yolov8_bmcv.py   | yolov8s-pose_int8_4b.bmodel        |      0.550      |    0.791   |
| SE9-8        | yolov8_bmcv.soc  | yolov8s-pose_fp32_1b.bmodel        |      0.574      |    0.793   |
| SE9-8        | yolov8_bmcv.soc  | yolov8s-pose_fp16_1b.bmodel        |      0.574      |    0.793   |
| SE9-8        | yolov8_bmcv.soc  | yolov8s-pose_int8_1b.bmodel        |      0.551      |    0.800   |
| SE9-8        | yolov8_bmcv.soc  | yolov8s-pose_int8_4b.bmodel        |      0.551      |    0.800   |


> **测试说明**：  
> 1. 由于sdk版本之间可能存在差异，实际运行结果与本表有<0.01的精度误差是正常的；
> 2. AP@IoU=0.5:0.95为area=all对应的指标。
> 3. 在搭载了相同TPU和SOPHONSDK的PCIe或SoC平台上，相同程序的精度一致，SE5系列对应BM1684，SE7系列对应BM1684X，SE9系列中，SE9-16对应BM1688，SE9-8对应CV186X；

## 7. 性能测试
### 7.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684/yolov8s-pose_fp32_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

|                测试模型                  | calculate time(ms) |
| ----------------------------------      | --------------- |
| BM1684/yolov8s-pose_fp32_1b.bmodel      |          27.65  |
| BM1684/yolov8s-pose_int8_1b.bmodel      |          16.12  |
| BM1684/yolov8s-pose_int8_4b.bmodel      |           7.73  |
| BM1684X/yolov8s-pose_fp32_1b.bmodel     |          31.02  |
| BM1684X/yolov8s-pose_fp16_1b.bmodel     |           5.68  |
| BM1684X/yolov8s-pose_int8_1b.bmodel     |           3.14  |
| BM1684X/yolov8s-pose_int8_4b.bmodel     |           2.83  |
| BM1688/yolov8s-pose_fp32_1b.bmodel      |         172.01  |
| BM1688/yolov8s-pose_fp16_1b.bmodel      |          35.40  |
| BM1688/yolov8s-pose_int8_1b.bmodel      |           7.83  |
| BM1688/yolov8s-pose_int8_4b.bmodel      |           7.53  |
| BM1688/yolov8s-pose_fp32_1b_2core.bmodel|          91.35  |
| BM1688/yolov8s-pose_fp16_1b_2core.bmodel|          20.67  |
| BM1688/yolov8s-pose_int8_1b_2core.bmodel|           6.93  |
| BM1688/yolov8s-pose_int8_4b_2core.bmodel|           4.78  |
| CV186X/yolov8s-pose_fp32_1b.bmodel      |         176.24  |
| CV186X/yolov8s-pose_fp16_1b.bmodel      |          37.41  |
| CV186X/yolov8s-pose_int8_1b.bmodel      |           9.36  |
| CV186X/yolov8s-pose_int8_4b.bmodel      |           9.12  |


> **测试说明**：  
> 1. 性能测试结果具有一定的波动性；
> 2. `calculate time`已折算为平均每张图片的推理时间；
> 3. SoC和PCIe的测试结果基本一致；

### 7.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++和Python例程打印的时间已经折算为单张图片的处理时间。

在不同的测试平台上，使用不同的例程、模型测试`datasets/coco/val2017_1000`，`conf_thresh=0.25, nms_thresh=0.7`，性能测试结果如下：

|    测试平台  |      测试程序       |               测试模型           |   decode_time   | preprocess_time | inference_time  |postprocess_time | 
| ----------- | ----------------- | ------------------------------- | --------------  | ------------    | -----------     | --------------  |
|   SE5-16    | yolov8_opencv.py  |   yolov8s-pose_fp32_1b.bmodel   |      7.65       |      21.60      |      32.32      |      2.47       |
|   SE5-16    | yolov8_opencv.py  |   yolov8s-pose_int8_1b.bmodel   |      6.83       |      21.76      |      20.85      |      2.45       |
|   SE5-16    | yolov8_opencv.py  |   yolov8s-pose_int8_4b.bmodel   |      6.84       |      23.48      |      11.73      |      2.23       |
|   SE5-16    |  yolov8_bmcv.py   |   yolov8s-pose_fp32_1b.bmodel   |      3.60       |      2.83       |      29.43      |      2.47       |
|   SE5-16    |  yolov8_bmcv.py   |   yolov8s-pose_int8_1b.bmodel   |      3.60       |      2.82       |      17.87      |      2.41       |
|   SE5-16    |  yolov8_bmcv.py   |   yolov8s-pose_int8_4b.bmodel   |      3.45       |      2.64       |      9.09       |      2.21       |
|   SE5-16    |  yolov8_bmcv.soc  |   yolov8s-pose_fp32_1b.bmodel   |      4.85       |      1.56       |      27.60      |      0.69       |
|   SE5-16    |  yolov8_bmcv.soc  |   yolov8s-pose_int8_1b.bmodel   |      4.97       |      1.57       |      16.07      |      0.69       |
|   SE5-16    |  yolov8_bmcv.soc  |   yolov8s-pose_int8_4b.bmodel   |      4.74       |      1.49       |      7.72       |      0.63       |
|   SE7-32    | yolov8_opencv.py  |   yolov8s-pose_fp32_1b.bmodel   |      17.72      |      25.24      |      37.31      |      2.57       |
|   SE7-32    | yolov8_opencv.py  |   yolov8s-pose_fp16_1b.bmodel   |      14.80      |      24.68      |      11.87      |      2.57       |
|   SE7-32    | yolov8_opencv.py  |   yolov8s-pose_int8_1b.bmodel   |      14.21      |      24.92      |      9.28       |      2.54       |
|   SE7-32    | yolov8_opencv.py  |   yolov8s-pose_int8_4b.bmodel   |      14.39      |      27.58      |      8.22       |      2.35       |
|   SE7-32    |  yolov8_bmcv.py   |   yolov8s-pose_fp32_1b.bmodel   |      3.39       |      2.54       |      33.44      |      2.59       |
|   SE7-32    |  yolov8_bmcv.py   |   yolov8s-pose_fp16_1b.bmodel   |      3.28       |      2.46       |      7.97       |      2.55       |
|   SE7-32    |  yolov8_bmcv.py   |   yolov8s-pose_int8_1b.bmodel   |      3.44       |      2.62       |      5.59       |      2.60       |
|   SE7-32    |  yolov8_bmcv.py   |   yolov8s-pose_int8_4b.bmodel   |      3.40       |      2.61       |      4.74       |      2.43       |
|   SE7-32    |  yolov8_bmcv.soc  |   yolov8s-pose_fp32_1b.bmodel   |      5.73       |      1.02       |      31.14      |      0.89       |
|   SE7-32    |  yolov8_bmcv.soc  |   yolov8s-pose_fp16_1b.bmodel   |      5.10       |      0.80       |      5.62       |      0.83       |
|   SE7-32    |  yolov8_bmcv.soc  |   yolov8s-pose_int8_1b.bmodel   |      5.04       |      0.80       |      3.09       |      0.80       |
|   SE7-32    |  yolov8_bmcv.soc  |   yolov8s-pose_int8_4b.bmodel   |      4.87       |      0.75       |      2.82       |      0.70       |
|   SE9-16    | yolov8_opencv.py  |   yolov8s-pose_fp32_1b.bmodel   |      9.42       |      29.95      |     178.69      |      3.48       |
|   SE9-16    | yolov8_opencv.py  |   yolov8s-pose_fp16_1b.bmodel   |      9.38       |      29.91      |      42.10      |      3.47       |
|   SE9-16    | yolov8_opencv.py  |   yolov8s-pose_int8_1b.bmodel   |      9.36       |      29.44      |      14.49      |      3.48       |
|   SE9-16    | yolov8_opencv.py  |   yolov8s-pose_int8_4b.bmodel   |      9.40       |      32.01      |      13.23      |      3.19       |
|   SE9-16    |  yolov8_bmcv.py   |   yolov8s-pose_fp32_1b.bmodel   |      4.32       |      4.93       |     174.68      |      3.47       |
|   SE9-16    |  yolov8_bmcv.py   |   yolov8s-pose_fp16_1b.bmodel   |      4.33       |      4.89       |      37.94      |      3.47       |
|   SE9-16    |  yolov8_bmcv.py   |   yolov8s-pose_int8_1b.bmodel   |      4.28       |      4.87       |      10.41      |      3.49       |
|   SE9-16    |  yolov8_bmcv.py   |   yolov8s-pose_int8_4b.bmodel   |      4.19       |      4.55       |      9.47       |      3.19       |
|   SE9-16    |  yolov8_bmcv.soc  |   yolov8s-pose_fp32_1b.bmodel   |      5.72       |      1.85       |     171.94      |      0.98       |
|   SE9-16    |  yolov8_bmcv.soc  |   yolov8s-pose_fp16_1b.bmodel   |      5.80       |      1.85       |      35.33      |      0.97       |
|   SE9-16    |  yolov8_bmcv.soc  |   yolov8s-pose_int8_1b.bmodel   |      5.73       |      1.84       |      7.74       |      0.96       |
|   SE9-16    |  yolov8_bmcv.soc  |   yolov8s-pose_int8_4b.bmodel   |      5.58       |      1.74       |      7.52       |      0.89       |
|   SE9-16    | yolov8_opencv.py  |yolov8s-pose_fp32_1b_2core.bmodel|      9.41       |      29.46      |      97.98      |      3.49       |
|   SE9-16    | yolov8_opencv.py  |yolov8s-pose_fp16_1b_2core.bmodel|      9.39       |      29.95      |      27.32      |      3.46       |
|   SE9-16    | yolov8_opencv.py  |yolov8s-pose_int8_1b_2core.bmodel|      9.37       |      29.81      |      13.65      |      3.49       |
|   SE9-16    | yolov8_opencv.py  |yolov8s-pose_int8_4b_2core.bmodel|      9.41       |      32.07      |      10.34      |      3.18       |
|   SE9-16    |  yolov8_bmcv.py   |yolov8s-pose_fp32_1b_2core.bmodel|      4.33       |      4.91       |      93.90      |      3.48       |
|   SE9-16    |  yolov8_bmcv.py   |yolov8s-pose_fp16_1b_2core.bmodel|      4.34       |      4.90       |      23.20      |      3.50       |
|   SE9-16    |  yolov8_bmcv.py   |yolov8s-pose_int8_1b_2core.bmodel|      4.31       |      4.88       |      9.49       |      3.50       |
|   SE9-16    |  yolov8_bmcv.py   |yolov8s-pose_int8_4b_2core.bmodel|      4.22       |      4.56       |      6.75       |      3.20       |
|   SE9-16    |  yolov8_bmcv.soc  |yolov8s-pose_fp32_1b_2core.bmodel|      5.81       |      1.83       |      91.25      |      0.97       |
|   SE9-16    |  yolov8_bmcv.soc  |yolov8s-pose_fp16_1b_2core.bmodel|      5.75       |      1.84       |      20.59      |      0.97       |
|   SE9-16    |  yolov8_bmcv.soc  |yolov8s-pose_int8_1b_2core.bmodel|      5.78       |      1.84       |      6.85       |      0.96       |
|   SE9-16    |  yolov8_bmcv.soc  |yolov8s-pose_int8_4b_2core.bmodel|      5.57       |      1.74       |      4.79       |      0.89       |
|    SE9-8    | yolov8_opencv.py  |   yolov8s-pose_fp32_1b.bmodel   |      9.42       |      29.57      |     182.99      |      3.49       |
|    SE9-8    | yolov8_opencv.py  |   yolov8s-pose_fp16_1b.bmodel   |      9.42       |      29.57      |      44.01      |      3.46       |
|    SE9-8    | yolov8_opencv.py  |   yolov8s-pose_int8_1b.bmodel   |      9.41       |      30.34      |      15.95      |      3.47       |
|    SE9-8    | yolov8_opencv.py  |   yolov8s-pose_int8_4b.bmodel   |      9.48       |      32.95      |      14.85      |      3.19       |
|    SE9-8    |  yolov8_bmcv.py   |   yolov8s-pose_fp32_1b.bmodel   |      4.14       |      4.59       |     179.02      |      3.49       |
|    SE9-8    |  yolov8_bmcv.py   |   yolov8s-pose_fp16_1b.bmodel   |      4.14       |      4.57       |      39.96      |      3.50       |
|    SE9-8    |  yolov8_bmcv.py   |   yolov8s-pose_int8_1b.bmodel   |      4.13       |      4.59       |      11.90      |      3.51       |
|    SE9-8    |  yolov8_bmcv.py   |   yolov8s-pose_int8_4b.bmodel   |      4.01       |      4.26       |      11.03      |      3.22       |
|    SE9-8    |  yolov8_bmcv.soc  |   yolov8s-pose_fp32_1b.bmodel   |      5.60       |      1.75       |     176.26      |      1.00       |
|    SE9-8    |  yolov8_bmcv.soc  |   yolov8s-pose_fp16_1b.bmodel   |      5.76       |      1.75       |      37.29      |      0.98       |
|    SE9-8    |  yolov8_bmcv.soc  |   yolov8s-pose_int8_1b.bmodel   |      5.66       |      1.75       |      9.26       |      0.98       |
|    SE9-8    |  yolov8_bmcv.soc  |   yolov8s-pose_int8_4b.bmodel   |      5.49       |      1.65       |      9.12       |      0.89       |
> **测试说明**：  
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. SE5-16/SE7-32的主控处理器均为8核CA53@2.3GHz，SE9-16为8核CA53@1.6GHz，SE9-8为6核CA53@1.6GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异，不同的阈值对后处理时间影响较大。 

## 8. FAQ
请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。

