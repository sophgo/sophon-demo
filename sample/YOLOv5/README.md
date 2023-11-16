[简体中文](./README.md) | [English](./README_EN.md)

# YOLOv5

## 目录

* [1. 简介](#1-简介)
* [2. 特性](#2-特性)
* [3. 准备模型与数据](#3-准备模型与数据)
* [4. 模型编译](#4-模型编译)
  * [4.1 TPU-NNTC编译BModel](#41-tpu-nntc编译bmodel)
  * [4.2 TPU-MLIR编译BModel](#42-tpu-mlir编译bmodel)
* [5. 例程测试](#5-例程测试)
* [6. 精度测试](#6-精度测试)
  * [6.1 测试方法](#61-测试方法)
  * [6.2 测试结果](#62-测试结果)
* [7. 性能测试](#7-性能测试)
  * [7.1 bmrt_test](#71-bmrt_test)
  * [7.2 程序运行性能](#72-程序运行性能)
* [8. YOLOv5 cpu opt](#8-yolov5-cpu-opt)
  * [8.1 NMS优化项](#81-nms优化项)
  * [8.2. 性能与精度测试](#82-性能与精度测试)
* [9. FAQ](#9-faq)
  
## 1. 简介
​YOLOv5是非常经典的基于anchor的One Stage目标检测算法，因其优秀的精度和速度表现，在工程实践应用中获得了非常广泛的应用。本例程对[​YOLOv5官方开源仓库](https://github.com/ultralytics/yolov5)v6.1版本的模型和算法进行移植，使之能在SOPHON BM1684\BM1684X\BM1688上进行推理测试。

## 2. 特性
* 支持BM1688(SoC)、BM1684X(x86 PCIe、SoC)、BM1684(x86 PCIe、SoC、arm PCIe)
* 支持FP32、FP16(BM1684X/BM1688)、INT8模型编译和推理
* 支持基于BMCV预处理的C++推理
* 支持基于OpenCV和BMCV预处理的Python推理
* 支持单batch和多batch模型推理
* 支持1个输出和3个输出模型推理
* 支持图片和视频测试
* 支持NMS后处理加速
 
## 3. 准备模型与数据
建议使用TPU-MLIR编译BModel，Pytorch模型在编译前要导出成onnx模型，如果您使用的tpu-mlir版本>=v1.3.0（即官网v23.07.01），可以直接使用torchscript模型。具体可参考[YOLOv5模型导出](./docs/YOLOv5_Export_Guide.md)。

​同时，您需要准备用于测试的数据集，如果量化模型，还要准备用于量化的数据集。

​本例程在`scripts`目录下提供了相关模型和数据的下载脚本`download.sh`，您也可以自己准备模型和数据集，并参考[4. 模型编译](#4-模型编译)进行模型转换。

```bash
# 安装unzip，若已安装请跳过，非ubuntu系统视情况使用yum或其他方式安装
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```

下载的模型包括：
```
./models
├── BM1684
│   ├── yolov5s_v6.1_3output_fp32_1b.bmodel   # 使用TPU-NNTC编译，用于BM1684的FP32 BModel，batch_size=1
│   ├── yolov5s_v6.1_3output_int8_1b.bmodel   # 使用TPU-NNTC编译，用于BM1684的INT8 BModel，batch_size=1
│   └── yolov5s_v6.1_3output_int8_4b.bmodel   # 使用TPU-NNTC编译，用于BM1684的INT8 BModel，batch_size=4
├── BM1684X
│   ├── yolov5s_v6.1_3output_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=1
│   ├── yolov5s_v6.1_3output_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=1
│   ├── yolov5s_v6.1_3output_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=1
│   └── yolov5s_v6.1_3output_int8_4b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=4
├── BM1688
│   ├── yolov5s_v6.1_3output_fp16_1b.bmodel       # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1
│   ├── yolov5s_v6.1_3output_fp32_1b.bmodel       # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1
│   ├── yolov5s_v6.1_3output_int8_1b.bmodel       # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=1
│   ├── yolov5s_v6.1_3output_int8_4b.bmodel       # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=1
│   ├── yolov5s_v6.1_3output_fp16_1b_2core.bmodel # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1, num_core=2
│   ├── yolov5s_v6.1_3output_fp32_1b_2core.bmodel # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1, num_core=2
│   ├── yolov5s_v6.1_3output_int8_1b_2core.bmodel # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=1, num_core=2
│   └── yolov5s_v6.1_3output_int8_4b_2core.bmodel # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=4, num_core=2
├── BM1684_ext                                # 相关单输出模型，此处没有benchmark，用户自行使用。
├── BM1684X_ext                               # 相关单输出模型，此处没有benchmark，用户自行使用。
│── torch
│   ├── yolov5m_v6.1_1output_torchscript.pt   # 相关单输出模型，此处没有benchmark，用户自行使用。
│   └── yolov5s_v6.1_3output.torchscript.pt   # trace后的torchscript模型
└── onnx
    ├── yolov5m_v6.1_1output_1b.onnx          # 相关单输出模型，此处没有benchmark，用户自行使用。
    ├── yolov5m_v6.1_1output_4b.onnx          # 相关单输出模型，此处没有benchmark，用户自行使用。
    └── yolov5s_v6.1_3output.onnx             # 导出的onnx动态模型       
```
下载的数据包括：
```
./datasets
├── test                                      # 测试图片
├── test_car_person_1080P.mp4                 # 测试视频
├── coco.names                                # coco类别名文件
├── coco128                                   # coco128数据集，用于模型量化
└── coco                                      
    ├── val2017_1000                          # coco val2017_1000数据集：coco val2017中随机抽取的1000张样本
    └── instances_val2017_1000.json           # coco val2017_1000数据集标签文件，用于计算精度评价指标  
```

## 4. 模型编译
导出的模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。建议使用TPU-MLIR编译BModel。

模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index/material/31/all.html)相应版本的SDK中获取)。

- 生成FP32 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684/BM1684X/BM1688**），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684 #bm1684x/bm1688
```

​执行上述命令会在`models/BM1684`等文件夹下生成`yolov5s_v6.1_3output_fp32_1b.bmodel`文件，即转换好的FP32 BModel。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688**），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x #bm1688
```

​执行上述命令会在`models/BM1684X/`等文件夹下生成`yolov5s_v6.1_3output_fp16_1b.bmodel`文件，即转换好的FP16 BModel。

- 生成INT8 BModel

​本例程在`scripts`目录下提供了量化INT8 BModel的脚本，请注意修改`gen_int8bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台（**支持BM1684/BM1684X/BM1688**），如：

```shell
./scripts/gen_int8bmodel_mlir.sh bm1684 #bm1684x/bm1688
```

​上述脚本会在`models/BM1684`等文件夹下生成`yolov5s_v6.1_3output_int8_1b.bmodel`等文件，即转换好的INT8 BModel。


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
python3 tools/eval_coco.py --gt_path datasets/coco/instances_val2017_1000.json --result_json results/yolov5s_v6.1_3output_fp32_1b.bmodel_val2017_1000_opencv_python_result.json
```
### 6.2 测试结果
CPP设置`--use_cpu_opt=false`或python不设置`--use_cpu_opt`进行测试，在coco2017val_1000数据集上，精度测试结果如下：
|   测试平台    |      测试程序     |              测试模型               |AP@IoU=0.5:0.95|AP@IoU=0.5|
| ------------ | ---------------- | ----------------------------------- | ------------- | -------- |
| BM1684 PCIe  | yolov5_opencv.py | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.377         | 0.580    |
| BM1684 PCIe  | yolov5_opencv.py | yolov5s_v6.1_3output_int8_1b.bmodel | 0.344         | 0.553    |
| BM1684 PCIe  | yolov5_bmcv.py   | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.373         | 0.573    |
| BM1684 PCIe  | yolov5_bmcv.py   | yolov5s_v6.1_3output_int8_1b.bmodel | 0.337         | 0.544    |
| BM1684 PCIe  | yolov5_bmcv.pcie | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.375         | 0.572    |
| BM1684 PCIe  | yolov5_bmcv.pcie | yolov5s_v6.1_3output_int8_1b.bmodel | 0.338         | 0.544    |
| BM1684 PCIe  | yolov5_sail.pcie | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.375         | 0.572    |
| BM1684 PCIe  | yolov5_sail.pcie | yolov5s_v6.1_3output_int8_1b.bmodel | 0.338         | 0.544    |
| BM1684X PCIe | yolov5_opencv.py | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.377         | 0.580    |
| BM1684X PCIe | yolov5_opencv.py | yolov5s_v6.1_3output_fp16_1b.bmodel | 0.377         | 0.580    |
| BM1684X PCIe | yolov5_opencv.py | yolov5s_v6.1_3output_int8_1b.bmodel | 0.363         | 0.572    |
| BM1684X PCIe | yolov5_bmcv.py   | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.373         | 0.573    |
| BM1684X PCIe | yolov5_bmcv.py   | yolov5s_v6.1_3output_fp16_1b.bmodel | 0.373         | 0.573    |
| BM1684X PCIe | yolov5_bmcv.py   | yolov5s_v6.1_3output_int8_1b.bmodel | 0.356         | 0.563    |
| BM1684X PCIe | yolov5_bmcv.pcie | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.374         | 0.572    |
| BM1684X PCIe | yolov5_bmcv.pcie | yolov5s_v6.1_3output_fp16_1b.bmodel | 0.374         | 0.572    |
| BM1684X PCIe | yolov5_bmcv.pcie | yolov5s_v6.1_3output_int8_1b.bmodel | 0.357         | 0.562    |
| BM1684X PCIe | yolov5_sail.pcie | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.374         | 0.572    |
| BM1684X PCIe | yolov5_sail.pcie | yolov5s_v6.1_3output_fp16_1b.bmodel | 0.374         | 0.572    |
| BM1684X PCIe | yolov5_sail.pcie | yolov5s_v6.1_3output_int8_1b.bmodel | 0.357         | 0.562    |
| BM1688 soc   | yolov5_bmcv.soc  | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.374         | 0.573    |
| BM1688 soc   | yolov5_bmcv.soc  | yolov5s_v6.1_3output_fp16_1b.bmodel | 0.374         | 0.573    |
| BM1688 soc   | yolov5_bmcv.soc  | yolov5s_v6.1_3output_int8_1b.bmodel | 0.353         | 0.564    |
| BM1688 soc   | yolov5_sail.soc  | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.374         | 0.573    |
| BM1688 soc   | yolov5_sail.soc  | yolov5s_v6.1_3output_fp16_1b.bmodel | 0.374         | 0.573    |
| BM1688 soc   | yolov5_sail.soc  | yolov5s_v6.1_3output_int8_1b.bmodel | 0.353         | 0.564    |
| BM1688 soc   | yolov5_opencv.py | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.378         | 0.579    |
| BM1688 soc   | yolov5_opencv.py | yolov5s_v6.1_3output_fp16_1b.bmodel | 0.377         | 0.579    |
| BM1688 soc   | yolov5_opencv.py | yolov5s_v6.1_3output_int8_1b.bmodel | 0.358         | 0.571    |
| BM1688 soc   | yolov5_bmcv.py   | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.374         | 0.573    |
| BM1688 soc   | yolov5_bmcv.py   | yolov5s_v6.1_3output_fp16_1b.bmodel | 0.374         | 0.573    |
| BM1688 soc   | yolov5_bmcv.py   | yolov5s_v6.1_3output_int8_1b.bmodel | 0.356         | 0.565    |
> **测试说明**：  
> 1. batch_size=4和batch_size=1的模型精度一致；
> 2. 由于sdk版本之间可能存在差异，实际运行结果与本表有<1%的精度误差是正常的；

## 7. 性能测试
### 7.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684/yolov5s_v6.1_3output_fp32_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

|                  测试模型                   | calculate time(ms) |
| ------------------------------------------- | ----------------- |
| BM1684/yolov5s_v6.1_3output_fp32_1b.bmodel  | 22.6              |
| BM1684/yolov5s_v6.1_3output_int8_1b.bmodel  | 11.5              |
| BM1684/yolov5s_v6.1_3output_int8_4b.bmodel  | 6.4               |
| BM1684X/yolov5s_v6.1_3output_fp32_1b.bmodel | 20.8              |
| BM1684X/yolov5s_v6.1_3output_fp16_1b.bmodel | 7.2               |
| BM1684X/yolov5s_v6.1_3output_int8_1b.bmodel | 3.5               |
| BM1684X/yolov5s_v6.1_3output_int8_4b.bmodel | 3.3               |

> **测试说明**：  
> 1. 性能测试结果具有一定的波动性；
> 2. `calculate time`已折算为平均每张图片的推理时间；
> 3. SoC和PCIe的测试结果基本一致。

### 7.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++例程打印的预处理时间、推理时间、后处理时间为整个batch处理的时间，需除以相应的batch size才是每张图片的处理时间。

CPP设置`--use_cpu_opt=false`或python不设置`--use_cpu_opt`进行测试，在不同的测试平台上，使用不同的例程、模型测试`datasets/coco/val2017_1000`，conf_thresh=0.5，nms_thresh=0.5，性能测试结果如下：
|    测试平台  |     测试程序      |             测试模型                |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ---------------- | ----------------------------------- | -------- | ---------     | ---------     | --------- |
| BM1684 SoC  | yolov5_opencv.py | yolov5s_v6.1_3output_fp32_1b.bmodel | 14.0     | 27.8          | 33.5          | 115       |
| BM1684 SoC  | yolov5_opencv.py | yolov5s_v6.1_3output_int8_1b.bmodel | 13.9     | 23.5          | 33.5          | 111       |
| BM1684 SoC  | yolov5_opencv.py | yolov5s_v6.1_3output_int8_4b.bmodel | 13.8     | 24.2          | 28.2          | 115       |
| BM1684 SoC  | yolov5_bmcv.py   | yolov5s_v6.1_3output_fp32_1b.bmodel | 3.0      | 3.0           | 28.5          | 111       |
| BM1684 SoC  | yolov5_bmcv.py   | yolov5s_v6.1_3output_int8_1b.bmodel | 3.0      | 2.4           | 17.4          | 111       |
| BM1684 SoC  | yolov5_bmcv.py   | yolov5s_v6.1_3output_int8_4b.bmodel | 2.8      | 2.3           | 11.5          | 115       |
| BM1684 SoC  | yolov5_bmcv.soc  | yolov5s_v6.1_3output_fp32_1b.bmodel | 5.4      | 1.5           | 22.6          | 35.6      |
| BM1684 SoC  | yolov5_bmcv.soc  | yolov5s_v6.1_3output_int8_1b.bmodel | 5.4      | 1.5           | 11.5          | 33.8      |
| BM1684 SoC  | yolov5_bmcv.soc  | yolov5s_v6.1_3output_int8_4b.bmodel | 5.2      | 1.6           | 6.2           | 33.1      |
| BM1684 SoC  | yolov5_sail.soc  | yolov5s_v6.1_3output_fp32_1b.bmodel | 3.3      | 3.1           | 23.3          | 34.6      |
| BM1684 SoC  | yolov5_sail.soc  | yolov5s_v6.1_3output_int8_1b.bmodel | 3.3      | 1.9           | 12.2          | 33.9      |
| BM1684 SoC  | yolov5_sail.soc  | yolov5s_v6.1_3output_int8_4b.bmodel | 3.1      | 1.8           | 6.9           | 33.2      |
| BM1684X SoC | yolov5_opencv.py | yolov5s_v6.1_3output_fp32_1b.bmodel | 15.0     | 22.4          | 32.0          | 104       |
| BM1684X SoC | yolov5_opencv.py | yolov5s_v6.1_3output_fp16_1b.bmodel | 15.0     | 22.4          | 18.5          | 104       |
| BM1684X SoC | yolov5_opencv.py | yolov5s_v6.1_3output_int8_1b.bmodel | 15.0     | 22.4          | 14.2          | 104       |
| BM1684X SoC | yolov5_opencv.py | yolov5s_v6.1_3output_int8_4b.bmodel | 14.9     | 23.1          | 14.5          | 108       |
| BM1684X SoC | yolov5_bmcv.py   | yolov5s_v6.1_3output_fp32_1b.bmodel | 3.1      | 2.4           | 28.8          | 104       |
| BM1684X SoC | yolov5_bmcv.py   | yolov5s_v6.1_3output_fp16_1b.bmodel | 3.1      | 2.4           | 15.5          | 104       |
| BM1684X SoC | yolov5_bmcv.py   | yolov5s_v6.1_3output_int8_1b.bmodel | 3.1      | 2.4           | 10.9          | 104       |
| BM1684X SoC | yolov5_bmcv.py   | yolov5s_v6.1_3output_int8_4b.bmodel | 2.9      | 2.3           | 9.8           | 109       |
| BM1684X SoC | yolov5_bmcv.soc  | yolov5s_v6.1_3output_fp32_1b.bmodel | 4.6      | 0.7           | 20.6          | 35.4      |
| BM1684X SoC | yolov5_bmcv.soc  | yolov5s_v6.1_3output_fp16_1b.bmodel | 4.6      | 0.7           | 7.1           | 35.4      |
| BM1684X SoC | yolov5_bmcv.soc  | yolov5s_v6.1_3output_int8_1b.bmodel | 4.6      | 0.7           | 3.4           | 34.3      |
| BM1684X SoC | yolov5_bmcv.soc  | yolov5s_v6.1_3output_int8_4b.bmodel | 4.4      | 0.7           | 3.2           | 34.0      |
| BM1684X SoC | yolov5_sail.soc  | yolov5s_v6.1_3output_fp32_1b.bmodel | 2.9      | 2.6           | 21.6          | 33.6      |
| BM1684X SoC | yolov5_sail.soc  | yolov5s_v6.1_3output_fp16_1b.bmodel | 2.9      | 2.6           | 8.1           | 33.6      |
| BM1684X SoC | yolov5_sail.soc  | yolov5s_v6.1_3output_int8_1b.bmodel | 2.9      | 2.6           | 4.3           | 32.4      |
| BM1684X SoC | yolov5_sail.soc  | yolov5s_v6.1_3output_int8_4b.bmodel | 2.6      | 2.6           | 4.0           | 32.0      |

> **测试说明**：  
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. BM1684/1684X SoC的主控处理器均为8核 ARM A53 42320 DMIPS @2.3GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异，不同的阈值对后处理时间影响较大。 

## 8. YOLOv5 cpu opt
本部分基于上述YOLOv5，优化了YOLOv5后处理NMS算法。下面主要说明NMS后处理算法优化的内容和优化后性能精度结果。

### 8.1. NMS优化项
* 提前噪声anchor的过滤，放在其他所有操作前，后续操作只需要处理数量显著减少的候选框
* 通过设置新阈值来优化掉anchor过滤中大量的sigmoid计算
* 优化存储减少数据遍历，在解码输出时仅仅保留候选框坐标、置信度、最高类别分数和对应索引
* 增大conf_thresh的值，过滤更多的噪声框
* 去除其他一些冗余计算

优化后NMS算法的时间瓶颈点在于模型输出的map大小，若尝试降低输出的map的高宽或通道数能够进一步降低NMS时间。
 
### 8.2. 性能与精度测试

在不同的测试平台上，使用不同的例程、模型测试`datasets/coco/val2017_1000`，conf_thresh=0.001，nms_thresh=0.6，cpp设置`--use_cpu_opt=true`或python设置`--use_cpu_opt`，NMS后处理算法改进前后性能和精度测试结果如下：
|    测试平台   |     测试程序      |             测试模型                | YOLOv5       | YOLOv5_cpu_opt|AP@IoU=0.5:0.95| 
| ------------ | ---------------- | ----------------------------------- | ------------ | ------------- | ------------- |
| BM1684 SoC   | yolov5_bmcv.soc  | yolov5s_v6.1_3output_fp32_1b.bmodel | 35.6         | 22.9          | 0.375         |
| BM1684 SoC   | yolov5_bmcv.soc  | yolov5s_v6.1_3output_int8_1b.bmodel | 33.8         | 20.5          | 0.339         |
| BM1684 SoC   | yolov5_sail.soc  | yolov5s_v6.1_3output_fp32_1b.bmodel | 34.6         | 21.1          | 0.375         |
| BM1684 SoC   | yolov5_sail.soc  | yolov5s_v6.1_3output_int8_1b.bmodel | 33.9         | 18.9          | 0.339         |
| BM1684 SoC   | yolov5_opencv.py | yolov5s_v6.1_3output_int8_1b.bmodel | 210.1        | 98.5          | 0.341         |
| BM1684 SoC   | yolov5_bmcv.py   | yolov5s_v6.1_3output_int8_1b.bmodel | 209.7        | 100.2         | 0.336         |

在不同的测试平台上，使用不同的例程、模型测试`datasets/coco/val2017_1000`，conf_thresh=0.01，nms_thresh=0.6，cpp设置`--use_cpu_opt=true`或python设置`--use_cpu_opt`，NMS后处理算法改进前后性能和精度测试结果如下：
|    测试平台   |     测试程序      |             测试模型                | YOLOv5       | YOLOv5_cpu_opt|AP@IoU=0.5:0.95|
| ------------ | ---------------- | ----------------------------------- | ------------ | ------------- | --------------|
| BM1684 SoC   | yolov5_bmcv.soc  | yolov5s_v6.1_3output_fp32_1b.bmodel | 18.1         | 7.5           | 0.373         |
| BM1684 SoC   | yolov5_bmcv.soc  | yolov5s_v6.1_3output_int8_1b.bmodel | 17.8         | 7.2           | 0.337         |
| BM1684 SoC   | yolov5_sail.soc  | yolov5s_v6.1_3output_fp32_1b.bmodel | 16.3         | 5.8           | 0.373         |
| BM1684 SoC   | yolov5_sail.soc  | yolov5s_v6.1_3output_int8_1b.bmodel | 16.0         | 5.5           | 0.337         |
| BM1684 SoC   | yolov5_opencv.py | yolov5s_v6.1_3output_int8_1b.bmodel | 118.8        | 23.0          | 0.339         |
| BM1684 SoC   | yolov5_bmcv.py   | yolov5s_v6.1_3output_int8_1b.bmodel | 116.5        | 23.1          | 0.334         |

> **注意：** 由于sail实现的方式与CPP保持一致，所以python调用后有轻微掉点，但速度有较大的提升。

若使用单类NMS，即设置`yolov5.cpp`文件中的宏`USE_MULTICLASS_NMS 0`或设置文件`yolov5_opencv.py`、`yolov5_bmcv.py`中的cpu opt接口的参数`input_use_multiclass_nms=False`和YOLOv5成员变量`multi_label=False`，在轻微损失精度的情况下能够提高后处理性能，使用不同的例程、模型测试`datasets/coco/val2017_1000`，conf_thresh=0.001，nms_thresh=0.6，cpp设置`--use_cpu_opt=true`或python设置`--use_cpu_opt`，NMS后处理算法改进前后性能和精度测试结果如下：
|    测试平台   |     测试程序      |             测试模型                | YOLOv5       | YOLOv5_cpu_opt|AP@IoU=0.5:0.95| 
| ------------ | ---------------- | ----------------------------------- | ------------ | ------------- | ------------- |
| BM1684 SoC   | yolov5_bmcv.soc  | yolov5s_v6.1_3output_fp32_1b.bmodel | 23.5         | 10.2          | 0.369         |
| BM1684 SoC   | yolov5_bmcv.soc  | yolov5s_v6.1_3output_int8_1b.bmodel | 23.1         | 9.9           | 0.332         |
| BM1684 SoC   | yolov5_sail.soc  | yolov5s_v6.1_3output_fp32_1b.bmodel | 21.6         | 8.5           | 0.369         |
| BM1684 SoC   | yolov5_sail.soc  | yolov5s_v6.1_3output_int8_1b.bmodel | 21.3         | 8.1           | 0.332         |
| BM1684 SoC   | yolov5_opencv.py | yolov5s_v6.1_3output_int8_1b.bmodel | 147.3        | 33.3          | 0.335         |
| BM1684 SoC   | yolov5_bmcv.py   | yolov5s_v6.1_3output_int8_1b.bmodel | 147.8        | 33.3          | 0.330         |

若使用单类NMS，即设置`yolov5.cpp`文件中的宏`USE_MULTICLASS_NMS 0`或设置文件`yolov5_opencv.py`、`yolov5_bmcv.py`中的cpu opt接口的参数`input_use_multiclass_nms=False`和YOLOv5成员变量`multi_label=False`，在轻微损失精度的情况下能够提高后处理性能，使用不同的例程、模型测试`datasets/coco/val2017_1000`，conf_thresh=0.01，nms_thresh=0.6，cpp设置`--use_cpu_opt=true`或python设置`--use_cpu_opt`，NMS后处理算法改进前后性能和精度测试结果如下：
|    测试平台   |     测试程序      |             测试模型                | YOLOv5       | YOLOv5_cpu_opt|AP@IoU=0.5:0.95|
| ------------ | ---------------- | ----------------------------------- | ------------ | ------------- | --------------|
| BM1684 SoC   | yolov5_bmcv.soc  | yolov5s_v6.1_3output_fp32_1b.bmodel | 17.6         | 6.2           | 0.367         |
| BM1684 SoC   | yolov5_bmcv.soc  | yolov5s_v6.1_3output_int8_1b.bmodel | 17.5         | 6.1           | 0.330         |
| BM1684 SoC   | yolov5_sail.soc  | yolov5s_v6.1_3output_fp32_1b.bmodel | 15.8         | 4.5           | 0.367         |
| BM1684 SoC   | yolov5_sail.soc  | yolov5s_v6.1_3output_int8_1b.bmodel | 15.7         | 4.3           | 0.330         |
| BM1684 SoC   | yolov5_opencv.py | yolov5s_v6.1_3output_int8_1b.bmodel | 114.7        | 9.7           | 0.333         |
| BM1684 SoC   | yolov5_bmcv.py   | yolov5s_v6.1_3output_int8_1b.bmodel | 114.2        | 9.6           | 0.327         |

> **测试说明**：  
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. BM1684/1684X SoC的主控处理器均为8核 ARM A53 42320 DMIPS @2.3GHz；
> 4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异，不同的阈值对后处理时间影响较大。 

## 9. FAQ
YOLOv5移植相关问题可参考[YOLOv5常见问题](./docs/YOLOv5_Common_Problems.md)，其他问题请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。