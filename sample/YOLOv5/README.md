# YOLOv5
## 目录
* [YOLOv5](#yolov5)
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
​YOLOv5是非常经典的基于anchor的One Stage目标检测算法，因其优秀的精度和速度表现，在工程实践应用中获得了非常广泛的应用。本例程对[​YOLOv5官方开源仓库](https://github.com/ultralytics/yolov5)v6.1版本的模型和算法进行移植，使之能在SOPHON BM1684和BM1684X上进行推理测试。

## 2. 特性
* 支持BM1684X(x86 PCIe、SoC)和BM1684(x86 PCIe、SoC)
* 支持FP32和INT8模型编译和推理
* 支持基于BMCV预处理的C++推理
* 支持基于OpenCV和BMCV预处理的Python推理
* 支持单batch和多batch模型推理
* 支持1个输出和多3个输出模型推理
* 支持图片和视频测试
 
## 3. 准备模型与数据
​Pytorch的模型在编译前要经过`torch.jit.trace`，trace后的模型才能用于编译BModel。可以使用官方仓库中的`export.py`导出torchscript模型，具体可参考[YOLOv5模型trace说明](./docs/YOLOv5_Export_Guide.md)。

​同时，您需要准备用于测试的数据集，如果量化模型，还要准备用于量化的数据集。

​本例程在`scripts`目录下提供了相关模型和数据的下载脚本`download.sh`，您也可以自己准备模型和数据集，并参考[4. 模型编译](#4-模型编译)进行模型转换。

```bash
chmod -R +x scripts/
./scripts/download.sh
```

下载的模型包括：
```
./models
├── BM1684
│   ├── yolov5s_v6.1_3output_fp32_1b.bmodel   # yolov5s，用于BM1684的FP32 BModel，batch_size=1，3个输出
│   ├── yolov5s_v6.1_3output_int8_1b.bmodel   # yolov5s，用于BM1684的INT8 BModel，batch_size=1，3个输出
│   └── yolov5s_v6.1_3output_int8_4b.bmodel   # yolov5s，用于BM1684的INT8 BModel，batch_size=4，3个输出
├── BM1684X
│   ├── yolov5s_v6.1_3output_fp32_1b.bmodel   # yolov5s，用于BM1684X的FP32 BModel，batch_size=1，3个输出
│   ├── yolov5s_v6.1_3output_int8_1b.bmodel   # yolov5s，用于BM1684X的INT8 BModel，batch_size=1，3个输出
│   └── yolov5s_v6.1_3output_int8_4b.bmodel   # yolov5s，用于BM1684X的INT8 BModel，batch_size=4，3个输出
└── torch
    └── yolov5s_v6.1_3output.torchscript.pt   # yolov5s，3个输出原始模型      
```
下载的数据包括：
```
./datasets
├── test                                      # 测试图片
├── test_car_person_1080P.mp4                 # 测试视频
├── coco.names                                # coco类别名文件
├── coco128                                   # coco128数据集，用于模型量化
└── coco                                      
    ├── val2017                               # coco val2017数据集
    └── instances_val2017.json                # coco val2017数据集标签文件，用于计算精度评价指标  
```

## 4. 模型编译
​trace后的pytorch模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。

模型编译前需要安装TPU-NNTC，具体可参考[tpu-nntc环境搭建](../docs/Environment_Install_Guide.md#1-tpu-nntc环境搭建)。安装好后需在tpu-nntc环境中进入例程目录。

### 4.1 生成FP32 BModel

pytorch模型编译为FP32 BModel，具体方法可参考《TPU-NNTC开发参考手册》的“BMNETP 使用”(请从[算能官网](https://developer.sophgo.com/site/index/material/28/all.html)相应版本的SDK中获取)。

​本例程在`scripts`目录下提供了编译FP32 BModel的脚本。请注意修改`gen_fp32bmodel.sh`中的JIT模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（支持BM1684和BM1684X），如：

```bash
./scripts/gen_fp32bmodel.sh BM1684
```

​执行上述命令会在`models/BM1684/`下生成`yolov5s_v6.1_3output_fp32_1b.bmodel`文件，即转换好的FP32 BModel。

### 4.2 生成INT8 BModel

​不量化模型可跳过本节。

pytorch模型的量化方法可参考《TPU-NNTC开发参考手册》的“模型量化”(请从[算能官网](https://developer.sophgo.com/site/index/material/28/all.html)相应版本的SDK中获取)，以及[模型量化注意事项](../../docs/Calibration_Guide.md#1-注意事项)。

​本例程在`scripts`目录下提供了量化INT8 BModel的脚本。请注意修改`gen_int8bmodel.sh`中的JIT模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台（支持BM1684和BM1684X），如：

```shell
./scripts/gen_int8bmodel.sh BM1684
```

​上述脚本会在`models/BM1684`下生成`yolov5s_v6.1_3output_int8_1b.bmodel`等文件，即转换好的INT8 BModel。

## 5. 例程测试
- [C++例程](./cpp/README.md)
- [Python例程](./python/README.md)

## 6. 精度测试
### 6.1 测试方法

首先，参考[C++例程](cpp/README.md#32-测试图片)或[Python例程](python/README.md#22-测试图片)推理要测试的数据集，生成预测的json文件，注意修改修改相关参数(conf_thresh=0.001、obj_thresh=0.001、nms_thresh=0.6)。  
然后，使用`tools`目录下的`eval_coco.py`脚本，将测试生成的json文件与测试集标签json文件进行对比，计算出目标检测的评价指标，命令如下：
```bash
# 安装pycocotools，若已安装请跳过
pip3 install pycocotools
# 请根据实际情况修改程序路径和json文件路径
python3 tools/eval_coco.py --label_json datasets/coco/instances_val2017.json --result_json python/results/pose_coco_fp32_1b.bmodel_val2017_opencv_python_result.json
```
### 6.2 测试结果
在coco2017val数据集上，精度测试结果如下：
|   测试平台    |      测试程序     |              测试模型               |AP@IoU=0.5:0.95|AR@IoU=0.5|
| ------------ | ---------------- | ----------------------------------- | ------------- | -------- |
| BM1684 PCIe  | yolov5_opencv.py | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.373         | 0.571    |
| BM1684 PCIe  | yolov5_opencv.py | yolov5s_v6.1_3output_int8_1b.bmodel | 0.342         | 0.542    |
| BM1684 PCIe  | yolov5_bmcv.py   | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.363         | 0.559    |
| BM1684 PCIe  | yolov5_bmcv.py   | yolov5s_v6.1_3output_int8_1b.bmodel | 0.330         | 0.526    |
| BM1684 PCIe  | yolov5_bmcv.pcie | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.359         | 0.562    |
| BM1684 PCIe  | yolov5_bmcv.pcie | yolov5s_v6.1_3output_int8_1b.bmodel | 0.332         | 0.534    |
| BM1684X PCIe | yolov5_opencv.py | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.373         | 0.571    |
| BM1684X PCIe | yolov5_opencv.py | yolov5s_v6.1_3output_int8_1b.bmodel | 0.356         | 0.554    |
| BM1684X PCIe | yolov5_bmcv.py   | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.363         | 0.559    |
| BM1684X PCIe | yolov5_bmcv.py   | yolov5s_v6.1_3output_int8_1b.bmodel | 0.345         | 0.540    |
| BM1684X PCIe | yolov5_bmcv.pcie | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.359         | 0.562    |
| BM1684X PCIe | yolov5_bmcv.pcie | yolov5s_v6.1_3output_int8_1b.bmodel | 0.345         | 0.546    |

> **测试说明**：  
1. batch_size=4和batch_size=1的模型精度一致；
2. SoC和PCIe的模型精度一致。

## 7. 性能测试
### 7.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径
bmrt_test --bmodel models/BM1684/yolov5s_v6.1_3output_fp32_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

|                  测试模型                   | calculate time(ms) |
| ------------------------------------------- | ----------------- |
| BM1684/yolov5s_v6.1_3output_fp32_1b.bmodel  | 22.6              |
| BM1684/yolov5s_v6.1_3output_int8_1b.bmodel  | 11.5              |
| BM1684/yolov5s_v6.1_3output_int8_4b.bmodel  | 6.4               |
| BM1684X/yolov5s_v6.1_3output_fp32_1b.bmodel | 20.5              |
| BM1684X/yolov5s_v6.1_3output_int8_1b.bmodel | 3.1               |
| BM1684X/yolov5s_v6.1_3output_int8_1b.bmodel | 2.9               |

> **测试说明**：  
1. 性能测试结果具有一定的波动性；
2. `calculate time`已折算为平均每张图片的推理时间。

### 7.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++例程打印的预处理时间、推理时间、后处理时间为整个batch处理的时间，需除以相应的batch size才是每张图片的处理时间。

在不同的测试平台上，使用不同的例程、模型测试`datasets/test`，性能测试结果如下：
|    测试平台   |     测试程序      |             测试模型                |decode_time|preprocess_time|inference_time|postprocess_time| 
| ------------ | ---------------- | ----------------------------------- | -------- | --------- | --------- | --------- |
| BM1684 PCIe  | yolov5_opencv.py | yolov5s_v6.1_3output_fp32_1b.bmodel | 6.5      | 7.5       | 34.5      | 15.0      |
| BM1684 PCIe  | yolov5_opencv.py | yolov5s_v6.1_3output_int8_1b.bmodel | 6.0      | 6.0       | 20.0      | 10.0      |
| BM1684 PCIe  | yolov5_opencv.py | yolov5s_v6.1_3output_int8_4b.bmodel | 5.5      | 5.5       | 15.8      | 10.3      |
| BM1684 PCIe  | yolov5_bmcv.py   | yolov5s_v6.1_3output_fp32_1b.bmodel | 1.5      | 2.2       | 30.5      | 15.8      |
| BM1684 PCIe  | yolov5_bmcv.py   | yolov5s_v6.1_3output_int8_1b.bmodel | 1.4      | 1.9       | 18.5      | 14.5      |
| BM1684 PCIe  | yolov5_bmcv.py   | yolov5s_v6.1_3output_int8_4b.bmodel | 1.4      | 1.9       | 11.8      | 11.8      |
| BM1684 PCIe  | yolov5_bmcv.pcie | yolov5s_v6.1_3output_fp32_1b.bmodel | 2.6      | 1.9       | 22.7      | 29.0      |
| BM1684 PCIe  | yolov5_bmcv.pcie | yolov5s_v6.1_3output_int8_1b.bmodel | 2.6      | 1.9       | 11.6      | 29.0      |
| BM1684 PCIe  | yolov5_bmcv.pcie | yolov5s_v6.1_3output_int8_4b.bmodel | 2.6      | 1.9       | 6.3       | 20.0      |
| BM1684 SoC   | yolov5_opencv.py | yolov5s_v6.1_3output_fp32_1b.bmodel | 22.8     | 28.2      | 33.7      | 114       |
| BM1684 SoC   | yolov5_opencv.py | yolov5s_v6.1_3output_int8_1b.bmodel | 22.0     | 24.0      | 33.5      | 111       |
| BM1684 SoC   | yolov5_opencv.py | yolov5s_v6.1_3output_int8_4b.bmodel | 22.0     | 24.5      | 28.5      | 115       |
| BM1684 SoC   | yolov5_bmcv.py   | yolov5s_v6.1_3output_fp32_1b.bmodel | 1.8      | 3.2       | 28.6      | 111       |
| BM1684 SoC   | yolov5_bmcv.py   | yolov5s_v6.1_3output_int8_1b.bmodel | 1.8      | 2.6       | 17.5      | 111       |
| BM1684 SoC   | yolov5_bmcv.py   | yolov5s_v6.1_3output_int8_4b.bmodel | 1.7      | 2.4       | 11.4      | 115       |
| BM1684 SoC   | yolov5_bmcv.pcie | yolov5s_v6.1_3output_fp32_1b.bmodel | 4.7      | 1.9       | 22.6      | 55.7      |
| BM1684 SoC   | yolov5_bmcv.pcie | yolov5s_v6.1_3output_int8_1b.bmodel | 4.7      | 1.9       | 11.5      | 55.9      |
| BM1684 SoC   | yolov5_bmcv.pcie | yolov5s_v6.1_3output_int8_4b.bmodel | 4.5      | 1.8       | 6.2       | 55.8      |
| BM1684X PCIe | yolov5_opencv.py | yolov5s_v6.1_3output_fp32_1b.bmodel | 5.8      | 4.5       | 24.5      | 7.5       |
| BM1684X PCIe | yolov5_opencv.py | yolov5s_v6.1_3output_int8_1b.bmodel | 5.6      | 4.5       | 9.0       | 7.5       |
| BM1684X PCIe | yolov5_opencv.py | yolov5s_v6.1_3output_int8_4b.bmodel | 5.6      | 4.5       | 9.1       | 9.5       |
| BM1684X PCIe | yolov5_bmcv.py   | yolov5s_v6.1_3output_fp32_1b.bmodel | 1.3      | 1.7       | 24.2      | 15.8      |
| BM1684X PCIe | yolov5_bmcv.py   | yolov5s_v6.1_3output_int8_1b.bmodel | 1.0      | 1.4       | 6.1       | 7.6       |
| BM1684X PCIe | yolov5_bmcv.py   | yolov5s_v6.1_3output_int8_4b.bmodel | 1.0      | 1.5       | 5.9       | 9.2       |
| BM1684X PCIe | yolov5_bmcv.pcie | yolov5s_v6.1_3output_fp32_1b.bmodel | 1.8      | 1.0       | 19.8      | 20.5      |
| BM1684X PCIe | yolov5_bmcv.pcie | yolov5s_v6.1_3output_int8_1b.bmodel | 1.7      | 0.8       | 3.1       | 15.0      |
| BM1684X PCIe | yolov5_bmcv.pcie | yolov5s_v6.1_3output_int8_4b.bmodel | 1.7      | 0.7       | 2.9       | 15.0      |
| BM1684X SoC  | yolov5_opencv.py | yolov5s_v6.1_3output_fp32_1b.bmodel | 21.8     | 23.5      | 30.4      | 105       |
| BM1684X SoC  | yolov5_opencv.py | yolov5s_v6.1_3output_int8_1b.bmodel | 21.8     | 23.5      | 25.5      | 104       |
| BM1684X SoC  | yolov5_opencv.py | yolov5s_v6.1_3output_int8_4b.bmodel | 21.7     | 23.2      | 24.6      | 108       |
| BM1684X SoC  | yolov5_bmcv.py   | yolov5s_v6.1_3output_fp32_1b.bmodel | 1.6      | 2.4       | 26.8      | 105       |
| BM1684X SoC  | yolov5_bmcv.py   | yolov5s_v6.1_3output_int8_1b.bmodel | 1.6      | 1.8       | 9.5       | 105       |
| BM1684X SoC  | yolov5_bmcv.py   | yolov5s_v6.1_3output_int8_4b.bmodel | 1.5      | 1.7       | 8.6       | 108       |
| BM1684X SoC  | yolov5_bmcv.pcie | yolov5s_v6.1_3output_fp32_1b.bmodel | 4.2      | 0.9       | 20.4      | 54.5      |
| BM1684X SoC  | yolov5_bmcv.pcie | yolov5s_v6.1_3output_int8_1b.bmodel | 4.2      | 0.7       | 3.1       | 54.5      |
| BM1684X SoC  | yolov5_bmcv.pcie | yolov5s_v6.1_3output_int8_4b.bmodel | 4.1      | 0.7       | 2.9       | 54.3      |

> **测试说明**：  
1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
2. 性能测试结果具有一定的波动性；
3. PCIe上的性能由于CPU的不同可能存在较大差异，BM1684 PCIe测试平台使用的CPU为Intel(R) Xeon(R) W-2255 CPU @ 3.70GHz，BM1684X PCIe测试平台使用的CPU为Intel(R) Core(TM) i5-7500 CPU @ 3.40GHz；
4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异。 