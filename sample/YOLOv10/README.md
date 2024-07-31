# YOLOv10

## 目录

- [YOLOv10](#yolov10)
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
    - [7.2 程序运行性能(待测)](#72-程序运行性能待测)
  - [8. FAQ](#8-faq)
  
## 1. 简介
​YOLOv10引入了一种新的实时目标检测方法，解决了YOLO 以前版本在后处理和模型架构方面的不足。通过消除非最大抑制（NMS）和优化各种模型组件，YOLOv10 在显著降低计算开销的同时实现了最先进的性能。本例程对[​YOLOv10官方开源仓库](https://github.com/THU-MIG/yolov10)的模型和算法进行移植，使之能在SOPHON BM1684X/BM1688/CV186X上进行推理测试。

## 2. 特性
* 支持BM1688/CV186X(SoC)、BM1684X(x86 PCIe、SoC)
* 支持FP32、FP16(BM1684X/BM1688/CV186X)、INT8模型编译和推理
* 支持基于BMCV预处理的C++推理
* 支持基于OpenCV和BMCV预处理的Python推理
* 支持单batch和多batch模型推理
* 支持1个输出模型推理
* C++例程支持opt模型推理（Python不支持）
* 支持图片和视频测试

## 3. 准备模型与数据
建议使用TPU-MLIR编译BModel，在使用TPU-MLIR编译前需要导出ONNX模型。具体可参考[YOLOv10模型导出](./docs/YOLOv10_Export_Guide.md)。

​同时，您需要准备用于测试的数据集，如果量化模型，还要准备用于量化的数据集。

​本例程在`scripts`目录下提供了相关模型和数据的下载脚本`download.sh`，您也可以自己准备模型和数据集，通过下载的mlir工具`tpu-mlir_v1.9.beta.0-116-g3c9d40a6d-20240720.tar.gz`，并参考[4. 模型编译](#4-模型编译)进行模型转换。

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
├── tpu-mlir_v1.9.beta.0-116-g3c9d40a6d-20240720.tar.gz # TPU-MLIR工具包	
├── BM1684X
│   ├── yolov10s_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=1  
│   ├── yolov10s_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=1 
│   ├── yolov10s_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=1
│   └── yolov10s_int8_4b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=4
├── BM1688
│   ├── yolov10s_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1, num_core=1
│   ├── yolov10s_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1, num_core=1
│   ├── yolov10s_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=1, num_core=1
│   ├── yolov10s_int8_4b.bmodel   # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=4, num_core=1
├── CV186X
│   ├── yolov10s_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于CV186X的FP32 BModel，batch_size=1
│   ├── yolov10s_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于CV186X的FP16 BModel，batch_size=1
│   ├── yolov10s_int8_1b.bmodel   # 使用TPU-MLIR编译，用于CV186X的INT8 BModel，batch_size=1
│   └── yolov10s_int8_4b.bmodel   # 使用TPU-MLIR编译，用于CV186X的INT8 BModel，batch_size=4
└── onnx
    ├── yolov10s.onnx      # 导出的动态opt onnx模型
    ├── yolov10s_qtable_mix       # TPU-MLIR编译时，用于BM1684X/BM1688的INT8 BModel混合精度量化
            
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
    └── instances_val2017_1000.json                # coco val2017_1000数据集关键点标签文件，用于计算精度评价指标 
```

## 4. 模型编译
导出的模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。若需要自行编译BModel，**建议使用前一节下载的TPU-MLIR编译BModel**。

模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)中1、2、3(3)步骤。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index/material/31/all.html)相应版本的SDK中获取)。
- 生成FP32 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684x #bm1684x/bm1688/cv186x
```

​执行上述命令会在`models/BM1684X`下生成`yolov10s_fp32_1b.bmodel`和`yolov10s_fp32_1b.bmodel`文件，即转换好的FP32 BModel。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x #bm1684x/bm1688/cv186x
```

​执行上述命令会在`models/BM1684X/`下生成`yolov10s_fp16_1b.bmodel`和`yolov10s_fp16_1b.bmodel`文件，即转换好的FP16 BModel。

- 生成INT8 BModel

​本例程在`scripts`目录下提供了量化INT8 BModel的脚本，请注意修改`gen_int8bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台（**支持BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_int8bmodel_mlir.sh bm1684x #bm1684x/bm1688/cv186x
```

​执行上述命令会在`models/BM1684X`下生成`yolov10s_int8_1b.bmodel`和`yolov10s_int8_1b.bmodel`等文件，即转换好的INT8 BModel。量化模型出现问题可以参考：[Calibration_Guide](../../docs/Calibration_Guide.md)。


## 5. 例程测试
- [C++例程](./cpp/README.md)
- [Python例程](./python/README.md)

## 6. 精度测试
### 6.1 测试方法

首先，参考[C++例程](cpp/README.md#32-测试图片)或[Python例程](python/README.md#22-测试图片)推理要测试的数据集，生成预测的json文件，注意修改数据集(datasets/coco/val2017_1000)和相关参数(conf_thresh=0.001)。  
然后，使用`tools`目录下的`eval_coco.py`脚本，将测试生成的json文件与测试集标签json文件进行对比，计算出目标检测的评价指标，命令如下：
```bash
# 安装pycocotools，若已安装请跳过
pip3 install pycocotools
# 请根据实际情况修改程序路径和json文件路径
python3 tools/eval_coco.py --gt_path datasets/coco/instances_val2017_1000.json --result_json results/yolov10s_fp32_1b.bmodel_val2017_1000_opencv_python_result.json
```
### 6.2 测试结果
在coco2017 val数据集上，精度测试结果如下：
| 测试平台 | 测试程序          | 测试模型                | AP@IoU=0.5:0.95 | AP@IoU=0.5 |
| -------- | ----------------- | ----------------------- | --------------- | ---------- |
| SE7-32   | yolov10_opencv.py | yolov10s_fp32_1b.bmodel | 0.463           | 0.627      |
| SE7-32   | yolov10_opencv.py | yolov10s_fp16_1b.bmodel | 0.452           | 0.610      |
| SE7-32   | yolov10_opencv.py | yolov10s_int8_1b.bmodel | 0.443           | 0.600      |
| SE7-32   | yolov10_opencv.py | yolov10s_int8_4b.bmodel | 0.443           | 0.600      |
| SE7-32   | yolov10_bmcv.py   | yolov10s_fp32_1b.bmodel | 0.462           | 0.627      |
| SE7-32   | yolov10_bmcv.py   | yolov10s_fp16_1b.bmodel | 0.452           | 0.609      |
| SE7-32   | yolov10_bmcv.py   | yolov10s_int8_1b.bmodel | 0.441           | 0.599      |
| SE7-32   | yolov10_bmcv.py   | yolov10s_int8_4b.bmodel | 0.441           | 0.599      |
| SE7-32   | yolov10_bmcv.soc  | yolov10s_fp32_1b.bmodel | 0.463           | 0.627      |
| SE7-32   | yolov10_bmcv.soc  | yolov10s_fp16_1b.bmodel | 0.453           | 0.610      |
| SE7-32   | yolov10_bmcv.soc  | yolov10s_int8_1b.bmodel | 0.441           | 0.598      |
| SE7-32   | yolov10_bmcv.soc  | yolov10s_int8_4b.bmodel | 0.441           | 0.598      |
| SE9-16   | yolov10_opencv.py | yolov10s_fp32_1b.bmodel | 0.463           | 0.627      |
| SE9-16   | yolov10_opencv.py | yolov10s_fp16_1b.bmodel | 0.452           | 0.609      |
| SE9-16   | yolov10_opencv.py | yolov10s_int8_1b.bmodel | 0.441           | 0.599      |
| SE9-16   | yolov10_opencv.py | yolov10s_int8_4b.bmodel | 0.441           | 0.599      |
| SE9-16   | yolov10_bmcv.py   | yolov10s_fp32_1b.bmodel | 0.462           | 0.627      |
| SE9-16   | yolov10_bmcv.py   | yolov10s_fp16_1b.bmodel | 0.452           | 0.609      |
| SE9-16   | yolov10_bmcv.py   | yolov10s_int8_1b.bmodel | 0.442           | 0.599      |
| SE9-16   | yolov10_bmcv.py   | yolov10s_int8_4b.bmodel | 0.442           | 0.599      |
| SE9-16   | yolov10_bmcv.soc  | yolov10s_fp32_1b.bmodel | 0.463           | 0.628      |
| SE9-16   | yolov10_bmcv.soc  | yolov10s_fp16_1b.bmodel | 0.453           | 0.610      |
| SE9-16   | yolov10_bmcv.soc  | yolov10s_int8_1b.bmodel | 0.443           | 0.600      |
| SE9-16   | yolov10_bmcv.soc  | yolov10s_int8_4b.bmodel | 0.443           | 0.600      |
| SE9-8    | yolov10_opencv.py | yolov10s_fp32_1b.bmodel | 0.463           | 0.627      |
| SE9-8    | yolov10_opencv.py | yolov10s_fp16_1b.bmodel | 0.453           | 0.610      |
| SE9-8    | yolov10_opencv.py | yolov10s_int8_1b.bmodel | 0.443           | 0.600      |
| SE9-8    | yolov10_opencv.py | yolov10s_int8_4b.bmodel | 0.443           | 0.600      |
| SE9-8    | yolov10_bmcv.py   | yolov10s_fp32_1b.bmodel | 0.462           | 0.627      |
| SE9-8    | yolov10_bmcv.py   | yolov10s_fp16_1b.bmodel | 0.452           | 0.609      |
| SE9-8    | yolov10_bmcv.py   | yolov10s_int8_1b.bmodel | 0.442           | 0.599      |
| SE9-8    | yolov10_bmcv.py   | yolov10s_int8_4b.bmodel | 0.442           | 0.599      |
| SE9-8    | yolov10_bmcv.soc  | yolov10s_fp32_1b.bmodel | 0.463           | 0.628      |
| SE9-8    | yolov10_bmcv.soc  | yolov10s_fp16_1b.bmodel | 0.453           | 0.610      |
| SE9-8    | yolov10_bmcv.soc  | yolov10s_int8_1b.bmodel | 0.443           | 0.600      |
| SE9-8    | yolov10_bmcv.soc  | yolov10s_int8_4b.bmodel | 0.441           | 0.660      |
> **测试说明**：  
> 1. batch_size=4和batch_size=1的模型精度一致；
> 2. 由于sdk版本之间可能存在差异，实际运行结果与本表有<0.01的精度误差是正常的；
> 3. AP@IoU=0.5:0.95为area=all对应的指标。
> 4. 在搭载了相同TPU和SOPHONSDK的PCIe或SoC平台上，相同程序的精度一致，SE7系列对应BM1684X，SE9系列中，SE9-16对应BM1688，SE9-8对应CV186X；


## 7. 性能测试
### 7.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684X/yolov10s_fp32_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

| 测试模型                        | calculate time(ms) |
| ------------------------------- | ------------------ |
| BM1684X/yolov10s_fp32_1b.bmodel | 22.42              |
| BM1684X/yolov10s_fp16_1b.bmodel | 6.52               |
| BM1684X/yolov10s_int8_1b.bmodel | 3.86               |
| BM1684X/yolov10s_int8_4b.bmodel | 3.28               |
| BM1688/yolov10s_fp32_1b.bmodel  | 131.20             |
| BM1688/yolov10s_fp16_1b.bmodel  | 35.65              |
| BM1688/yolov10s_int8_1b.bmodel  | 10.23              |
| BM1688/yolov10s_int8_4b.bmodel  | 9.18               |
| CV186X/yolov10s_fp32_1b.bmodel  | 131.12             |
| CV186X/yolov10s_fp16_1b.bmodel  | 35.65              |
| CV186X/yolov10s_int8_1b.bmodel  | 10.25              |
| CV186X/yolov10s_int8_4b.bmodel  | 9.15               |
> **测试说明**：  
1. 性能测试结果具有一定的波动性；
2. `calculate time`已折算为平均每张图片的推理时间；
3. SoC和PCIe的测试结果基本一致。


### 7.2 程序运行性能(待测)
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++和Python例程打印的时间已经折算为单张图片的处理时间。

在不同的测试平台上，使用不同的例程、模型测试`datasets/val2017_1000`，conf_thresh=0.001，性能测试结果如下：
| 测试平台 | 测试程序          | 测试模型                | decode_time | preprocess_time | inference_time | postprocess_time |
| -------- | ----------------- | ----------------------- | ----------- | --------------- | -------------- | ---------------- |
| SE7-32   | yolov10_opencv.py | yolov10s_fp32_1b.bmodel | 14.77       | 22.81           | 26.16          | 0.61             |
| SE7-32   | yolov10_opencv.py | yolov10s_fp16_1b.bmodel | 13.74       | 22.48           | 10.22          | 0.61             |
| SE7-32   | yolov10_opencv.py | yolov10s_int8_1b.bmodel | 13.74       | 22.47           | 7.81           | 0.61             |
| SE7-32   | yolov10_opencv.py | yolov10s_int8_4b.bmodel | 13.82       | 24.79           | 6.58           | 0.42             |
| SE7-32   | yolov10_bmcv.py   | yolov10s_fp32_1b.bmodel | 3.00        | 2.25            | 22.57          | 0.56             |
| SE7-32   | yolov10_bmcv.py   | yolov10s_fp16_1b.bmodel | 3.00        | 2.25            | 6.64           | 0.56             |
| SE7-32   | yolov10_bmcv.py   | yolov10s_int8_1b.bmodel | 2.98        | 2.24            | 4.25           | 0.55             |
| SE7-32   | yolov10_bmcv.py   | yolov10s_int8_4b.bmodel | 2.84        | 2.09            | 3.39           | 0.38             |
| SE7-32   | yolov10_bmcv.soc  | yolov10s_fp32_1b.bmodel | 4.25        | 0.74            | 22.02          | 0.025            |
| SE7-32   | yolov10_bmcv.soc  | yolov10s_fp16_1b.bmodel | 4.30        | 0.74            | 6.08           | 0.025            |
| SE7-32   | yolov10_bmcv.soc  | yolov10s_int8_1b.bmodel | 4.30        | 0.74            | 3.69           | 0.025            |
| SE7-32   | yolov10_bmcv.soc  | yolov10s_int8_4b.bmodel | 4.18        | 0.71            | 3.25           | 0.012            |
| SE9-16   | yolov10_opencv.py | yolov10s_fp32_1b.bmodel | 4.48        | 29.99           | 136.02         | 0.84             |
| SE9-16   | yolov10_opencv.py | yolov10s_fp16_1b.bmodel | 4.48        | 30.07           | 30.07          | 0.84             |
| SE9-16   | yolov10_opencv.py | yolov10s_int8_1b.bmodel | 4.45        | 29.65           | 15.19          | 0.83             |
| SE9-16   | yolov10_opencv.py | yolov10s_int8_4b.bmodel | 4.55        | 33.85           | 13.18          | 0.58             |
| SE9-16   | yolov10_bmcv.py   | yolov10s_fp32_1b.bmodel | 4.39        | 4.93            | 131.64         | 0.79             |
| SE9-16   | yolov10_bmcv.py   | yolov10s_fp16_1b.bmodel | 4.38        | 4.90            | 36.22          | 0.81             |
| SE9-16   | yolov10_bmcv.py   | yolov10s_int8_1b.bmodel | 4.31        | 4.87            | 10.77          | 0.81             |
| SE9-16   | yolov10_bmcv.py   | yolov10s_int8_4b.bmodel | 4.21        | 4.59            | 9.30           | 0.54             |
| SE9-16   | yolov10_bmcv.soc  | yolov10s_fp32_1b.bmodel | 5.76        | 1.83            | 35.26          | 0.04             |
| SE9-16   | yolov10_bmcv.soc  | yolov10s_fp16_1b.bmodel | 5.73        | 1.83            | 9.84           | 0.04             |
| SE9-16   | yolov10_bmcv.soc  | yolov10s_int8_1b.bmodel | 5.73        | 1.83            | 9.84           | 0.04             |
| SE9-16   | yolov10_bmcv.soc  | yolov10s_int8_4b.bmodel | 5.57        | 1.75            | 9.06           | 0.02             |
| SE9-8    | yolov10_opencv.py | yolov10s_fp32_1b.bmodel | 23.86       | 29.78           | 135.99         | 0.86             |
| SE9-8    | yolov10_opencv.py | yolov10s_fp16_1b.bmodel | 23.13       | 29.58           | 40.68          | 0.86             |
| SE9-8    | yolov10_opencv.py | yolov10s_int8_1b.bmodel | 19.17       | 30.22           | 15.20          | 0.85             |
| SE9-8    | yolov10_opencv.py | yolov10s_int8_4b.bmodel | 19.26       | 32.95           | 13.15          | 0.60             |
| SE9-8    | yolov10_bmcv.py   | yolov10s_fp32_1b.bmodel | 4.12        | 4.58            | 131.57         | 0.80             |
| SE9-8    | yolov10_bmcv.py   | yolov10s_fp16_1b.bmodel | 4.11        | 4.58            | 36.15          | 0.81             |
| SE9-8    | yolov10_bmcv.py   | yolov10s_int8_1b.bmodel | 4.08        | 4.56            | 10.70          | 0.80             |
| SE9-8    | yolov10_bmcv.py   | yolov10s_int8_4b.bmodel | 3.90        | 4.26            | 9.24           | 0.55             |
| SE9-8    | yolov10_bmcv.soc  | yolov10s_fp32_1b.bmodel | 5.59        | 1.74            | 130.46         | 0.039            |
| SE9-8    | yolov10_bmcv.soc  | yolov10s_fp16_1b.bmodel | 5.62        | 1.73            | 35.22          | 0.04             |
| SE9-8    | yolov10_bmcv.soc  | yolov10s_int8_1b.bmodel | 5.60        | 1.74            | 9.79           | 0.04             |
| SE9-8    | yolov10_bmcv.soc  | yolov10s_int8_4b.bmodel | 5.40        | 1.65            | 9.00           | 0.02             |
> **测试说明**：  
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. SE7-32的主控处理器均为8核CA53@2.3GHz，SE9-16为8核CA53@1.6GHz，SE9-8为6核CA53@1.6GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异，不同的阈值对后处理时间影响较大。 


## 8. FAQ
请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。