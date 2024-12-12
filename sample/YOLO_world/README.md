# YOLO_world

## 目录

- [YOLO\_world](#yolo_world)
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
YOLO_world，作为腾讯人工智能实验室的又一力作，不仅继承了YOLO系列模型在实时性方面的优势，更在开放词汇检测方面取得了重大突破。它采用了视觉语言建模和预训练的方法，能够在无需预先训练的情况下，实时识别图像中任何由描述性文本指定的物体。本例程对[​yoloworld官方开源仓库](https://github.com/ultralytics/ultralytics)的模型和算法进行移植，使之能在SOPHON BM1684X/BM1688/CV186X上进行推理测试。

## 2. 特性
* 支持BM1688/CV186X(SoC)、BM1684X(x86 PCIe、SoC)
* 支持FP32、FP16(BM1684X/BM1688/CV186X)、INT8模型编译和推理
* 支持基于OpenCV和BMCV预处理的Python推理
* 支持1个输出模型推理
* 支持图片和视频测试

## 3. 准备模型与数据
建议使用TPU-MLIR编译BModel，在使用TPU-MLIR编译前需要导出ONNX模型。具体可参考[YOLO_world模型导出](./docs/YOLO_World_Export_Guide.md)。

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
├── BM1684X
│   ├── yoloworld_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=1
│   ├── yoloworld_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=1
│   ├── yoloworld_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=1
│   └── clip_text_vitb32_bm1684x_f16_1b.bmodel          # encode_text部分fp16 bmodel 
├── BM1688
│   ├── yoloworld_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1, num_core=1
│   ├── yoloworld_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1, num_core=1
│   ├── yoloworld_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=1, num_core=1
│   ├── yoloworld_fp32_1b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1, num_core=2
│   ├── yoloworld_fp16_1b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1, num_core=2
│   ├── yoloworld_int8_1b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=1, num_core=2
│   ├── clip_text_vitb32_bm1688_f16_1b_2core.bmodel     # encode_text部分fp16 bmodel，num_core=2
│   └── clip_text_vitb32_bm1688_f16_1b.bmodel           # encode_text部分fp16 bmodel
├── CV186X
│   ├── yoloworld_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于CV186X的FP32 BModel，batch_size=1
│   ├── yoloworld_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于CV186X的FP16 BModel，batch_size=1
│   ├── yoloworld_int8_1b.bmodel   # 使用TPU-MLIR编译，用于CV186X的INT8 BModel，batch_size=1
│   └── clip_text_vitb32_cv186x_f16_1b.bmodel           # encode_text部分fp16 bmodel
├── onnx
│   ├── yoloworld.onnx      # 导出的onnx模型
│   ├── coco128_npz        # coco128量化数据集
│   └── clip_text_vitb32.onnx                           # encode_text部分onnx模型    
│ 
├── bpe_simple_vocab_16e6.txt.gz                        # 提供BPE分词所需的合并规则和基础词汇表
└── text_projection_512_512.npy                         # 导出encode_text onnx模型时保存的text_projection数据，在bmodel推理时使用
    
    
         
```
下载的数据包括：
```
./datasets
├── test                                      # 测试图片
├── test_car_person_1080P.mp4                 # 测试视频
├── coco.names                                # coco类别名文件
├── coco128                                   # coco128 图片数据集
└── coco                                      
    ├── val2017_1000                               # coco val2017_1000数据集：coco val2017中随机抽取的1000张样本
    └── instances_val2017_1000.json                # coco val2017_1000数据集关键点标签文件，用于计算精度评价指标 
```

## 4. 模型编译
导出的模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。建议使用TPU-MLIR编译BModel。

模型编译前需要安装TPU-MLIR，本例程使用的TPU-MLIR版本是`v1.6`，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的SDK中获取)。

- 生成FP32 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684x #bm1688/cv186x
```

​执行上述命令会在`models/BM1684X`下生成`yoloworld_fp32_1b.bmodel`文件，即转换好的FP32 BModel。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x #bm1688/cv186x
```

​执行上述命令会在`models/BM1684X/`下生成`yoloworld_fp16_1b.bmodel`文件，即转换好的FP16 BModel。

- 生成INT8 BModel

​本例程在`scripts`目录下提供了量化INT8 BModel的脚本，请注意修改`gen_int8bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台（**支持BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_int8bmodel_mlir.sh bm1684x #bm1684x/bm1688/cv186x
```

​执行上述命令会在`models/BM1684X`下生成`yoloworld_int8_1b.bmodel`等文件，即转换好的INT8 BModel。量化模型出现问题可以参考：[Calibration_Guide](../../docs/Calibration_Guide.md)。


## 5. 例程测试
- [Python例程](./python/README.md)

## 6. 精度测试
### 6.1 测试方法

首先，参考[Python例程](python/README.md#22-测试图片)推理要测试的数据集，生成预测的json文件，注意修改数据集(datasets/coco/val2017_1000)和相关参数(class_names="all"、conf_thresh=0.001、nms_thresh=0.7)。  
然后，使用`tools`目录下的`eval_coco.py`脚本，将测试生成的json文件与测试集标签json文件进行对比，计算出目标检测的评价指标，命令如下：
```bash
# 安装pycocotools，若已安装请跳过
pip3 install pycocotools
# 请根据实际情况修改程序路径和json文件路径
python3 tools/eval_coco.py --gt_path datasets/coco/instances_val2017_1000.json --result_json results/yoloworld_fp32_1b.bmodel_val2017_1000_opencv_python_result.json
```
### 6.2 测试结果
在coco2017 val数据集上，精度测试结果如下：
|   测试平台    |      测试程序     |      测试模型          |AP@IoU=0.5:0.95|AP@IoU=0.5|
| ------------ | ---------------- | ---------------------- | ------------- | -------- |
| SE7-32       | yoloworld_opencv.py | yoloworld_fp32_1b.bmodel |    0.370 |    0.514 |
| SE7-32       | yoloworld_opencv.py | yoloworld_fp16_1b.bmodel |    0.370 |    0.514 |
| SE7-32       | yoloworld_opencv.py | yoloworld_int8_1b.bmodel |    0.349 |    0.494 |
| SE7-32       | yoloworld_bmcv.py | yoloworld_fp32_1b.bmodel |    0.370 |    0.515 |
| SE7-32       | yoloworld_bmcv.py | yoloworld_fp16_1b.bmodel |    0.371 |    0.514 |
| SE7-32       | yoloworld_bmcv.py | yoloworld_int8_1b.bmodel |    0.349 |    0.495 |
| SE9-16       | yoloworld_opencv.py | yoloworld_fp32_1b.bmodel |    0.370 |    0.514 |
| SE9-16       | yoloworld_opencv.py | yoloworld_fp16_1b.bmodel |    0.370 |    0.514 |
| SE9-16       | yoloworld_opencv.py | yoloworld_int8_1b.bmodel |    0.349 |    0.495 |
| SE9-16       | yoloworld_bmcv.py | yoloworld_fp32_1b.bmodel |    0.370 |    0.515 |
| SE9-16       | yoloworld_bmcv.py | yoloworld_fp16_1b.bmodel |    0.371 |    0.514 |
| SE9-16       | yoloworld_bmcv.py | yoloworld_int8_1b.bmodel |    0.350 |    0.494 |
| SE9-16       | yoloworld_opencv.py | yoloworld_fp32_1b_2core.bmodel |    0.365 |    0.511 |
| SE9-16       | yoloworld_opencv.py | yoloworld_fp16_1b_2core.bmodel |    0.370 |    0.514 |
| SE9-16       | yoloworld_opencv.py | yoloworld_int8_1b_2core.bmodel |    0.349 |    0.495 |
| SE9-16       | yoloworld_bmcv.py | yoloworld_fp32_1b_2core.bmodel |    0.365 |    0.512 |
| SE9-16       | yoloworld_bmcv.py | yoloworld_fp16_1b_2core.bmodel |    0.371 |    0.514 |
| SE9-16       | yoloworld_bmcv.py | yoloworld_int8_1b_2core.bmodel |    0.350 |    0.494 |
| SE9-8       | yoloworld_opencv.py | yoloworld_fp32_1b.bmodel |    0.370 |    0.514 |
| SE9-8       | yoloworld_opencv.py | yoloworld_fp16_1b.bmodel |    0.370 |    0.514 |
| SE9-8       | yoloworld_opencv.py | yoloworld_int8_1b.bmodel |    0.349 |    0.495 |
| SE9-8       | yoloworld_bmcv.py | yoloworld_fp32_1b.bmodel |    0.370 |    0.515 |
| SE9-8       | yoloworld_bmcv.py | yoloworld_fp16_1b.bmodel |    0.371 |    0.514 |
| SE9-8       | yoloworld_bmcv.py | yoloworld_int8_1b.bmodel |    0.350 |    0.494 |
> **测试说明**：  
> 1. batch_size=4和batch_size=1的模型精度一致；
> 2. 由于sdk版本之间可能存在差异，实际运行结果与本表有<0.01的精度误差是正常的；
> 3. AP@IoU=0.5:0.95为area=all对应的指标。
> 4. 在搭载了相同TPU和SOPHONSDK的PCIe或SoC平台上，相同程序的精度一致，SE5系列对应BM1684，SE7系列对应BM1684X，SE9系列中，SE9-16对应BM1688，SE9-8对应CV186X；


## 7. 性能测试
### 7.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684X/yoloworld_fp32_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

|              测试模型               | calculate time(ms) |
| ----------------------------------- | ----------------- |
| BM1684X/yoloworld_fp32_1b.bmodel   |          35.69  |
| BM1684X/yoloworld_fp16_1b.bmodel   |           7.50  |
| BM1684X/yoloworld_int8_1b.bmodel   |           5.02  |
| BM1688/yoloworld_fp32_1b.bmodel    |         184.76  |
| BM1688/yoloworld_fp16_1b.bmodel    |          38.99  |
| BM1688/yoloworld_int8_1b.bmodel    |          16.30  |
| BM1688/yoloworld_fp32_1b_2core.bmodel|          99.15  |
| BM1688/yoloworld_fp16_1b_2core.bmodel|          24.38  |
| BM1688/yoloworld_int8_1b_2core.bmodel|          12.71  |
| CV186X/yoloworld_fp32_1b.bmodel    |         184.80  |
| CV186X/yoloworld_fp16_1b.bmodel    |          39.03  |
| CV186X/yoloworld_int8_1b.bmodel    |          16.36  |
> **测试说明**：  
1. 性能测试结果具有一定的波动性；
2. `calculate time`已折算为平均每张图片的推理时间；
3. SoC和PCIe的测试结果基本一致。


### 7.2 程序运行性能
参考[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++和Python例程打印的时间已经折算为单张图片的处理时间。

在不同的测试平台上，使用不同的例程、模型测试`datasets/val2017_1000`，conf_thresh=0.25，nms_thresh=0.7，性能测试结果如下：
|    测试平台  |     测试程序      |        测试模型        |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ---------------- | ---------------------- | --------  | ---------    | ---------     | ---------      |
|   SE7-32    |yoloworld_opencv.py|yoloworld_fp32_1b.bmodel |      9.56       |      22.52      |      41.79      |      4.48       |
|   SE7-32    |yoloworld_opencv.py|yoloworld_fp16_1b.bmodel |      7.41       |      22.83      |      13.55      |      4.50       |
|   SE7-32    |yoloworld_opencv.py|yoloworld_int8_1b.bmodel |      6.79       |      22.67      |      11.08      |      4.47       |
|   SE7-32    | yoloworld_bmcv.py |yoloworld_fp32_1b.bmodel |      3.03       |      2.30       |      44.70      |      4.52       |
|   SE7-32    | yoloworld_bmcv.py |yoloworld_fp16_1b.bmodel |      3.02       |      2.29       |      16.37      |      4.52       |
|   SE7-32    | yoloworld_bmcv.py |yoloworld_int8_1b.bmodel |      3.05       |      2.29       |      14.01      |      4.48       |
|   SE9-16    |yoloworld_opencv.py|yoloworld_fp32_1b.bmodel |      19.44      |      29.54      |     192.39      |      5.66       |
|   SE9-16    |yoloworld_opencv.py|yoloworld_fp16_1b.bmodel |      12.02      |      30.10      |      46.74      |      5.65       |
|   SE9-16    |yoloworld_opencv.py|yoloworld_int8_1b.bmodel |      9.41       |      29.94      |      23.96      |      5.66       |
|   SE9-16    | yoloworld_bmcv.py |yoloworld_fp32_1b.bmodel |      4.23       |      4.71       |     195.66      |      5.66       |
|   SE9-16    | yoloworld_bmcv.py |yoloworld_fp16_1b.bmodel |      4.24       |      4.73       |      49.85      |      5.67       |
|   SE9-16    | yoloworld_bmcv.py |yoloworld_int8_1b.bmodel |      4.23       |      4.72       |      27.92      |      5.67       |
|   SE9-16    |yoloworld_opencv.py|yoloworld_fp32_1b_2core.bmodel|      9.42       |      29.65      |     106.97      |      5.62       |
|   SE9-16    |yoloworld_opencv.py|yoloworld_fp16_1b_2core.bmodel|      9.42       |      29.96      |      31.96      |      5.63       |
|   SE9-16    |yoloworld_opencv.py|yoloworld_int8_1b_2core.bmodel|      9.39       |      30.01      |      20.31      |      5.64       |
|   SE9-16    | yoloworld_bmcv.py |yoloworld_fp32_1b_2core.bmodel|      4.26       |      4.72       |     110.11      |      5.68       |
|   SE9-16    | yoloworld_bmcv.py |yoloworld_fp16_1b_2core.bmodel|      4.21       |      4.73       |      35.43      |      5.65       |
|   SE9-16    | yoloworld_bmcv.py |yoloworld_int8_1b_2core.bmodel|      4.24       |      4.71       |      23.66      |      5.68       |
|    SE9-8    |yoloworld_opencv.py|yoloworld_fp32_1b.bmodel |      13.81      |      30.34      |     192.50      |      5.73       |
|    SE9-8    |yoloworld_opencv.py|yoloworld_fp16_1b.bmodel |      14.24      |      29.72      |      46.66      |      5.71       |
|    SE9-8    |yoloworld_opencv.py|yoloworld_int8_1b.bmodel |      13.24      |      29.68      |      24.00      |      5.71       |
|    SE9-8    | yoloworld_bmcv.py |yoloworld_fp32_1b.bmodel |      7.84       |      4.62       |     195.78      |      5.75       |
|    SE9-8    | yoloworld_bmcv.py |yoloworld_fp16_1b.bmodel |      7.40       |      4.60       |      50.11      |      5.72       |
|    SE9-8    | yoloworld_bmcv.py |yoloworld_int8_1b.bmodel |      7.46       |      4.58       |      27.35      |      5.73       |

> **测试说明**：  
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. SE7-32的主控处理器为8核CA53@2.3GHz，SE9-16为8核CA53@1.6GHz，SE9-8为6核CA53@1.6GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异，不同的阈值对后处理时间影响较大。 


## 8. FAQ
请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。