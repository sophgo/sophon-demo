# YOLO26

## 目录

- [YOLO26](#yolo26)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
  - [3. 数据准备与模型编译](#3-数据准备与模型编译)
    - [3.1 数据准备](#31-数据准备)
    - [3.2 模型编译](#32-模型编译)
  - [4. 例程测试](#4-例程测试)
  - [5. 精度测试](#5-精度测试)
    - [5.1 测试方法](#51-测试方法)
    - [5.2 测试结果](#52-测试结果)
  - [6. 性能测试](#6-性能测试)
    - [6.1 bmrt\_test](#61-bmrt_test)
    - [6.2 程序运行性能](#62-程序运行性能)
  - [8. FAQ](#8-faq)

## 1. 简介
​YOLO26针对边缘部署进行了优化，具有端到端无 NMS 推理功能。本例程对[​YOLO26官方开源仓库](https://github.com/ultralytics/ultralytics)v8.4.9的模型和算法进行移植，使之能在SOPHON BM1684X/BM1688/CV186X上进行推理测试。

## 2. 特性
* 支持BM1688/CV186X(SoC)、BM1684X(x86 PCIe、SoC)
* 支持FP32、FP16、INT8模型编译和推理
* 支持基于BMCV预处理的C++推理
* 支持基于OpenCV和BMCV预处理的Python推理
* 支持端到端无 NMS 推理
* 支持图片和视频测试

## 3. 数据准备与模型编译
### 3.1 数据准备
​本例程在`scripts`目录下提供了相关模型和数据的下载脚本`download.sh`，**如果您希望自己准备模型和数据集，可以跳过本小节，参考[3.2 模型编译](#32-模型编译)进行模型转换。**


```bash
# 安装unzip，若已安装请跳过，非ubuntu系统视情况使用yum或其他方式安装
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```

`download.sh`默认只下载`datasets`，`models`可以通过指定参数分平台下载，参数如下：
```bash
--all     # 下载所有模型
--BM1684X # 下载BM1684X的bmodel
--BM1688  # 下载BM1688的bmodel
--CV186X  # 下载CV186X的bmodel
--onnx    # 下载onnx
```

```
下载的模型包括：
./models
├── BM1684X # 在BM1684X上运行的模型
│   ├── yolo26s_fp32_1b.bmodel
│   ├── yolo26s_fp16_1b.bmodel 
│   └── yolo26s_int8_1b.bmodel
├── BM1688 # 在BM1688上运行的模型
│   ├── yolo26s_fp32_1b.bmodel
│   ├── yolo26s_fp16_1b.bmodel
│   ├── yolo26s_int8_1b.bmodel
│   └── yolo26s_int8_1b_2core.bmodel
├── CV186X # 在CV186X上运行的模型
│   ├── yolo26s_fp32_1b.bmodel
│   ├── yolo26s_fp16_1b.bmodel
│   └── yolo26s_int8_1b.bmodel
└── onnx
    ├── yolo26s.onnx             # 导出的onnx模型
    ├── yolo26s_qtable_f16       # 编译F16 BModel需要混合精度的层
    └── yolo26s_qtable_int8      # 量化INT8 BModel需要混合精度的层
            
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
    └── instances_val2017_1000.json                # coco val2017_1000数据集标签文件，用于计算精度评价指标 
```

## 4. 模型编译
**如果您不编译模型，只想直接使用下载的数据集和模型，可以跳过本小节。**

源模型需要编译成BModel才能在SOPHON TPU上运行，源模型在编译前要导出成onnx模型，具体可参考[YOLO26模型导出](./docs/YOLO26_Export_Guide.md)。​同时，您需要准备用于测试的数据集，如果量化模型，还要准备用于量化的数据集。

建议使用TPU-MLIR编译BModel，模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录，并使用本例程提供的脚本将onnx模型编译为BModel。脚本中命令的详细说明可参考《TPU-MLIR开发手册》(请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的SDK中获取)。

- 生成FP32 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684x #bm1688/cv186x
```

​执行上述命令会在`models/BM1684X`文件夹下生成转换好的FP32 BModel。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x #bm1688/cv186x
```

​执行上述命令会在`models/BM1684X/`文件夹下生成转换好的FP16 BModel。

- 生成INT8 BModel

​本例程在`scripts`目录下提供了量化INT8 BModel的脚本，请注意修改`gen_int8bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台（**支持BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_int8bmodel_mlir.sh bm1684x #bm1688/cv186x
```

​上述脚本会在`models/BM1684X`文件夹下生成转换好的INT8 BModel。

注：这里用到了混合精度量化，需要将一些层设为敏感层，相应的qtable在此前`download.sh`下载的`models/onnx`文件夹里。如果您需要量化自己微调过的模型，可以参考[量化指南](./docs/YOLO26_Calibration_Guide.md)中的方法。

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
python3 tools/eval_coco.py --gt_path datasets/coco/instances_val2017_1000.json --result_json results/yolo26s_fp32_1b.bmodel_val2017_1000_opencv_python_result.json
```
### 6.2 测试结果
在coco2017 val数据集上，精度测试结果如下：
| 测试平台 | 测试程序          | 测试模型                | AP@IoU=0.5:0.95 | AP@IoU=0.5 |
| -------- | ----------------- | ----------------------- | --------------- | ---------- |
| SE7-32   | yolo26_opencv.py | yolo26s_fp32_1b.bmodel | 0.480           | 0.643      |
| SE7-32   | yolo26_opencv.py | yolo26s_fp16_1b.bmodel | 0.480           | 0.643      |
| SE7-32   | yolo26_opencv.py | yolo26s_int8_1b.bmodel | 0.471           | 0.640      |
| SE7-32   | yolo26_bmcv.py   | yolo26s_fp32_1b.bmodel | 0.480           | 0.643      |
| SE7-32   | yolo26_bmcv.py   | yolo26s_fp16_1b.bmodel | 0.481           | 0.643      |
| SE7-32   | yolo26_bmcv.py   | yolo26s_int8_1b.bmodel | 0.470           | 0.638      |
| SE7-32   | yolo26_bmcv.soc  | yolo26s_fp32_1b.bmodel | 0.480           | 0.643      |
| SE7-32   | yolo26_bmcv.soc  | yolo26s_fp16_1b.bmodel | 0.480           | 0.643      |
| SE7-32   | yolo26_bmcv.soc  | yolo26s_int8_1b.bmodel | 0.470           | 0.637      |
| SE9-16   | yolo26_opencv.py | yolo26s_fp32_1b.bmodel | 0.480           | 0.643      |
| SE9-16   | yolo26_opencv.py | yolo26s_fp16_1b.bmodel | 0.480           | 0.643      |
| SE9-16   | yolo26_opencv.py | yolo26s_int8_1b.bmodel | 0.471           | 0.640      |
| SE9-16   | yolo26_opencv.py | yolo26s_int8_1b_2core.bmodel | 0.471           | 0.640      |
| SE9-16   | yolo26_bmcv.py   | yolo26s_fp32_1b.bmodel | 0.480           | 0.643      |
| SE9-16   | yolo26_bmcv.py   | yolo26s_fp16_1b.bmodel | 0.481           | 0.643      |
| SE9-16   | yolo26_bmcv.py   | yolo26s_int8_1b.bmodel | 0.470           | 0.638      |
| SE9-16   | yolo26_bmcv.py   | yolo26s_int8_1b_2core.bmodel | 0.470           | 0.638      |
| SE9-16   | yolo26_bmcv.soc  | yolo26s_fp32_1b.bmodel | 0.480           | 0.643      |
| SE9-16   | yolo26_bmcv.soc  | yolo26s_fp16_1b.bmodel | 0.480           | 0.643      |
| SE9-16   | yolo26_bmcv.soc  | yolo26s_int8_1b.bmodel | 0.471           | 0.638      |
| SE9-16   | yolo26_bmcv.soc  | yolo26s_int8_1b_2core.bmodel | 0.471           | 0.638      |
|    SE9-8    | yolo26_opencv.py  |      yolo26s_fp32_1b.bmodel       | 0.480 | 0.643 |
|    SE9-8    | yolo26_opencv.py  |      yolo26s_fp16_1b.bmodel       | 0.480 | 0.643 |
|    SE9-8    | yolo26_opencv.py  |      yolo26s_int8_1b.bmodel       | 0.471 | 0.640 |
|    SE9-8    |  yolo26_bmcv.py   |      yolo26s_fp32_1b.bmodel       | 0.480 | 0.643 |
|    SE9-8    |  yolo26_bmcv.py   |      yolo26s_fp16_1b.bmodel       | 0.481 | 0.643 |
|    SE9-8    |  yolo26_bmcv.py   |      yolo26s_int8_1b.bmodel       | 0.470 | 0.638 |
|    SE9-8    |  yolo26_bmcv.soc  |      yolo26s_fp32_1b.bmodel       | 0.480 | 0.643 |
|    SE9-8    |  yolo26_bmcv.soc  |      yolo26s_fp16_1b.bmodel       | 0.480 | 0.643 |
|    SE9-8    |  yolo26_bmcv.soc  |      yolo26s_int8_1b.bmodel       | 0.471 | 0.638 |


> **测试说明**：  
> 1. 由于sdk版本之间可能存在差异，实际运行结果与本表有<0.01的精度误差是正常的；
> 2. AP@IoU=0.5:0.95为area=all对应的指标。
> 3. 在搭载了相同TPU和SOPHONSDK的PCIe或SoC平台上，相同程序的精度一致，SE7系列对应BM1684X，SE9系列中，SE9-16对应BM1688，SE9-8对应CV186X；


## 7. 性能测试
### 7.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684X/yolo26s_fp32_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

|    测试平台  | 测试模型                        | calculate time(ms) |
| ----------- | ------------------------------- | ------------------ |
|   SE7-32    | BM1684X/yolo26s_fp32_1b.bmodel | 26.54              |
|   SE7-32    | BM1684X/yolo26s_fp16_1b.bmodel | 7.29               |
|   SE7-32    | BM1684X/yolo26s_int8_1b.bmodel | 4.89               |
|   SE9-16    | BM1688/yolo26s_fp32_1b.bmodel  | 130.88             |
|   SE9-16    | BM1688/yolo26s_fp16_1b.bmodel  | 35.62              |
|   SE9-16    | BM1688/yolo26s_int8_1b.bmodel  | 13.62              |
|   SE9-16    | BM1688/yolo26s_int8_1b_2core.bmodel  | 10.80              |
|   SE9-8     | CV186X/yolo26s_fp32_1b.bmodel  | 130.85             |
|   SE9-8     | CV186X/yolo26s_fp16_1b.bmodel  | 35.62              |
|   SE9-8     | CV186X/yolo26s_int8_1b.bmodel  | 13.66              |

> **测试说明**：  
1. 性能测试结果具有一定的波动性；
2. `calculate time`已折算为平均每张图片的推理时间；
3. SoC和PCIe的测试结果基本一致。


### 7.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++和Python例程打印的时间已经折算为单张图片的处理时间。

在不同的测试平台上，使用不同的例程、模型测试`datasets/val2017_1000`，conf_thresh=0.001，性能测试结果如下：
| 测试平台 | 测试程序          | 测试模型                | decode_time | preprocess_time | inference_time | postprocess_time |
| -------- | ----------------- | ----------------------- | ----------- | --------------- | -------------- | ---------------- |
| SE7-32   | yolo26_opencv.py | yolo26s_fp32_1b.bmodel | 6.79       | 22.91           | 29.77          | 0.60             |
| SE7-32   | yolo26_opencv.py | yolo26s_fp16_1b.bmodel | 6.80       | 22.94           | 10.22          | 0.60             |
| SE7-32   | yolo26_opencv.py | yolo26s_int8_1b.bmodel | 6.80       | 22.93           | 7.97           | 0.61             |
| SE7-32   | yolo26_bmcv.py   | yolo26s_fp32_1b.bmodel | 2.91        | 1.99            | 26.12          | 0.58             |
| SE7-32   | yolo26_bmcv.py   | yolo26s_fp16_1b.bmodel | 2.90        | 1.98            | 6.88           | 0.57             |
| SE7-32   | yolo26_bmcv.py   | yolo26s_int8_1b.bmodel | 2.92        | 1.99            | 4.46           | 0.57             |
| SE7-32   | yolo26_bmcv.soc  | yolo26s_fp32_1b.bmodel | 4.44        | 0.76            | 25.85          | 0.12            |
| SE7-32   | yolo26_bmcv.soc  | yolo26s_fp16_1b.bmodel | 4.44        | 0.76            | 6.59           | 0.12            |
| SE7-32   | yolo26_bmcv.soc  | yolo26s_int8_1b.bmodel | 4.46        | 0.76            | 4.18           | 0.12            |
| SE9-16   | yolo26_opencv.py | yolo26s_fp32_1b.bmodel | 9.36        | 35.17           | 135.31         | 0.63             |
| SE9-16   | yolo26_opencv.py | yolo26s_fp16_1b.bmodel | 9.27        | 35.35           | 39.86          | 0.63             |
| SE9-16   | yolo26_opencv.py | yolo26s_int8_1b.bmodel | 9.17        | 34.49           | 17.83          | 0.63             |
| SE9-16   | yolo26_opencv.py | yolo26s_int8_1b_2core.bmodel | 9.20        | 34.67           | 15.03          | 0.63             |
| SE9-16   | yolo26_bmcv.py   | yolo26s_fp32_1b.bmodel | 3.87        | 3.99            | 130.84         | 0.61             |
| SE9-16   | yolo26_bmcv.py   | yolo26s_fp16_1b.bmodel | 3.86        | 3.98            | 35.54          | 0.61             |
| SE9-16   | yolo26_bmcv.py   | yolo26s_int8_1b.bmodel | 3.85        | 3.99            | 13.54          | 0.61             |
| SE9-16   | yolo26_bmcv.py   | yolo26s_int8_1b_2core.bmodel | 3.87        | 3.99            | 10.72          | 0.61             |
| SE9-16   | yolo26_bmcv.soc  | yolo26s_fp32_1b.bmodel | 5.89        | 1.74            | 130.34         | 0.17             |
| SE9-16   | yolo26_bmcv.soc  | yolo26s_fp16_1b.bmodel | 5.87        | 1.74            | 35.09          | 0.17             |
| SE9-16   | yolo26_bmcv.soc  | yolo26s_int8_1b.bmodel | 5.90        | 1.76            | 13.12          | 0.18             |
| SE9-16   | yolo26_bmcv.soc  | yolo26s_int8_1b_2core.bmodel | 5.88        | 1.74            | 10.28          | 0.17             |
|    SE9-8    | yolo26_opencv.py  |      yolo26s_fp32_1b.bmodel       |      9.54       |      34.69      |     134.96      |      0.63       |
|    SE9-8    | yolo26_opencv.py  |      yolo26s_fp16_1b.bmodel       |      9.55       |      34.77      |      39.72      |      0.62       |
|    SE9-8    | yolo26_opencv.py  |      yolo26s_int8_1b.bmodel       |      9.53       |      34.73      |      17.74      |      0.63       |
|    SE9-8    |  yolo26_bmcv.py   |      yolo26s_fp32_1b.bmodel       |      3.90       |      3.99       |     130.73      |      0.58       |
|    SE9-8    |  yolo26_bmcv.py   |      yolo26s_fp16_1b.bmodel       |      3.83       |      3.98       |      35.47      |      0.57       |
|    SE9-8    |  yolo26_bmcv.py   |      yolo26s_int8_1b.bmodel       |      3.84       |      3.99       |      13.47      |      0.57       |
|    SE9-8    |  yolo26_bmcv.soc  |      yolo26s_fp32_1b.bmodel       |      5.64       |      1.74       |     130.23      |      0.09       |
|    SE9-8    |  yolo26_bmcv.soc  |      yolo26s_fp16_1b.bmodel       |      5.78       |      1.74       |      35.02      |      0.09       |
|    SE9-8    |  yolo26_bmcv.soc  |      yolo26s_int8_1b.bmodel       |      5.71       |      1.74       |      13.04      |      0.09       |



> **测试说明**：  
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. SE7-32的主控处理器均为8核CA53@2.3GHz，SE9-16为8核CA53@1.6GHz，SE9-8为6核CA53@1.6GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异，不同的阈值对后处理时间影响较大。 


## 8. FAQ
请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。