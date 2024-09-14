# YOLOv8_seg

## 目录

- [YOLOv8\_seg](#yolov8_seg)
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
​YOLOv8_seg是YOLO系列的的一个重大更新版本，它抛弃了以往的YOLO系类模型使用的Anchor-Base，采用了Anchor-Free的思想。YOLOv8建立在YOLO系列成功的基础上，通过对网络结构的改造，进一步提升其性能和灵活性。本例程对[​YOLOv8官方开源仓库](https://github.com/ultralytics/ultralytics)的模型和算法进行移植，使之能在SOPHON BM1684/BM1684X/BM1688上进行推理测试。

## 2. 特性
* 支持BM1688(SoC)和BM1684X(x86 PCIe、SoC)和BM1684(x86 PCIe、SoC、arm PCIe)
* 支持FP32、FP16(BM1684X/BM1688)、INT8模型编译和推理
* 支持基于BMCV预处理的C++推理
* 支持基于OpenCV和BMCV预处理的Python推理
* 支持单batch和多batch模型推理
* 支持1个输出模型推理
* 支持图片和视频测试

## 3. 准备模型与数据
建议使用TPU-MLIR编译BModel，在使用TPU-MLIR编译前需要导出ONNX模型。具体可参考[YOLOv8模型导出](./docs/YOLOv8_Export_Guide.md)。如需导出后处理TPU加速模型，可参考[MaskBmodel模型导出](./docs/MaskBmodel_Export_Guide.md)。

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
|   ├── yolov8s_getmask_32_fp32.bmodel # 使用TPU-MLIR编译，用于BM1684的后处理TPU加速模型
│   ├── yolov8s_fp32_1b.bmodel         # 使用TPU-MLIR编译，用于BM1684的FP32 BModel，batch_size=1
│   ├── yolov8s_int8_1b.bmodel         # 使用TPU-MLIR编译，用于BM1684的INT8 BModel，batch_size=1
│   └── yolov8s_int8_4b.bmodel         # 使用TPU-MLIR编译，用于BM1684的INT8 BModel，batch_size=4
├── BM1684X
|   ├── yolov8s_getmask_32_fp32.bmodel # 使用TPU-MLIR编译，用于BM1684X的后处理TPU加速模型
│   ├── yolov8s_fp32_1b.bmodel         # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=1
│   ├── yolov8s_fp16_1b.bmodel         # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=1
│   ├── yolov8s_int8_1b.bmodel         # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=1
│   └── yolov8s_int8_4b.bmodel         # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=4
├── BM1688
|   ├── yolov8s_getmask_32_fp32.bmodel # 使用TPU-MLIR编译，用于BM1688的后处理TPU加速模型
│   ├── yolov8s_fp32_1b.bmodel         # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1，num_core=1
│   ├── yolov8s_fp16_1b.bmodel         # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1，num_core=1
│   ├── yolov8s_int8_1b.bmodel         # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=1，num_core=1
│   ├── yolov8s_int8_4b.bmodel         # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=4，num_core=1
│   ├── yolov8s_fp32_1b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1，num_core=2
│   ├── yolov8s_fp16_1b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1，num_core=2
│   ├── yolov8s_int8_1b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=1，num_core=2
│   └── yolov8s_int8_4b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=4，num_core=2
├── onnx
│   ├── yolov8s_getmask_32_fp32.onnx   # 用于编译后处理TPU加速模型的onnx
│   ├── yolov8s-seg-1b.onnx            # 导出的静态onnx模型，batch_size=1
│   ├── yolov8s-seg-4b.onnx            # 导出的静态onnx模型，batch_size=4
│   ├── yolov8s_seg_bm1684_qtable      # TPU-MLIR编译时，用于BM1684的INT8 BModel混合精度量化
│   ├── yolov8s_seg_bm1684x_qtable     # TPU-MLIR编译时，用于BM1684X的INT8 BModel混合精度量化
│   └── yolov8s_seg_bm1688_qtable      # TPU-MLIR编译时，用于BM1688的INT8 BModel混合精度量化
│── torch
    └── yolov8s-seg.pt   # trace后的torchscript模型
    
         
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
导出的模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。建议使用TPU-MLIR编译BModel。

模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的SDK中获取)。

- 生成FP32 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684/BM1684X/BM1688**），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684 #bm1688
```

​执行上述命令会在`models/BM1684`下生成`yolov8s_fp32_1b.bmodel`文件，即转换好的FP32 BModel。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688**），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x #bm1688
```

​执行上述命令会在`models/BM1684X/`下生成`yolov8s_fp16_1b.bmodel`文件，即转换好的FP16 BModel。

- 生成INT8 BModel

​本例程在`scripts`目录下提供了量化INT8 BModel的脚本，请注意修改`gen_int8bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台（**支持BM1684/BM1684X/BM1688**），如：

```bash
./scripts/gen_int8bmodel_mlir.sh bm1684 #bm1684x/bm1688

```

​执行上述命令会在`models/BM1684`下生成`yolov8s_int8_1b.bmodel`等文件，即转换好的INT8 BModel。

## 5. 例程测试
- [C++例程](./cpp/README.md)
- [Python例程](./python/README.md)

## 6. 精度测试
### 6.1 测试方法

首先，参考[C++例程](cpp/README.md#32-测试图片)或[Python例程](python/README.md#22-测试图片)推理要测试的数据集，生成预测的json文件，注意修改数据集(datasets/coco/val2017_1000)和相关参数(conf_thresh=0.001、nms_thresh=0.7)。  
然后，使用`tools`目录下的`eval_coco.py`脚本，将测试生成的json文件与测试集标签json文件进行对比，计算出目标检测的评价指标，命令如下：
```bash
# 安装pycocotools，若已安装请跳过
pip3 install pycocotools
# 请根据实际情况修改程序路径和json文件路径
python3 tools/eval_coco.py --gt_path datasets/coco/instances_val2017_1000.json --result_json results/yolov8s_fp32_1b.bmodel_val2017_1000_opencv_python_result.json
```
### 6.2 测试结果
在coco2017 val数据集上，精度测试结果如下：
|   测试平台    |      测试程序     |      测试模型          |AP@IoU=0.5:0.95|AP@IoU=0.5|
| ------------ | ---------------- | ---------------------- | ------------- | -------- |
| SE5-16       | yolov8_opencv.py | yolov8s_fp32_1b.bmodel | 0.357   | 0.569 |
| SE5-16       | yolov8_opencv.py | yolov8s_int8_1b.bmodel | 0.346   | 0.545 |
| SE5-16       | yolov8_bmcv.py   | yolov8s_fp32_1b.bmodel | 0.357   | 0.568 |
| SE5-16       | yolov8_bmcv.py   | yolov8s_int8_1b.bmodel | 0.346   | 0.545 |
| SE5-16       | yolov8_bmcv.soc | yolov8s_fp32_1b.bmodel | 0.350   | 0.566 |
| SE5-16       | yolov8_bmcv.soc | yolov8s_int8_1b.bmodel | 0.342   | 0.551 |
| SE7-32       | yolov8_opencv.py | yolov8s_fp32_1b.bmodel | 0.358   | 0.569 |
| SE7-32       | yolov8_opencv.py | yolov8s_fp16_1b.bmodel | 0.358   | 0.569 |
| SE7-32       | yolov8_opencv.py | yolov8s_int8_1b.bmodel | 0.353   | 0.555 |
| SE7-32       | yolov8_bmcv.py   | yolov8s_fp32_1b.bmodel | 0.357   | 0.569 |
| SE7-32       | yolov8_bmcv.py   | yolov8s_fp16_1b.bmodel | 0.357   | 0.569 |
| SE7-32       | yolov8_bmcv.py   | yolov8s_int8_1b.bmodel | 0.352   | 0.557 |
| SE7-32       | yolov8_bmcv.soc | yolov8s_fp32_1b.bmodel | 0.351   | 0.567 |
| SE7-32       | yolov8_bmcv.soc | yolov8s_fp16_1b.bmodel | 0.352   | 0.568 |
| SE7-32       | yolov8_bmcv.soc | yolov8s_int8_1b.bmodel | 0.349   | 0.563 |
| SE9-16       | yolov8_opencv.py | yolov8s_fp32_1b.bmodel       |    0.358 |    0.569 |
| SE9-16       | yolov8_opencv.py | yolov8s_fp16_1b.bmodel       |    0.358 |    0.569 |
| SE9-16       | yolov8_opencv.py | yolov8s_int8_1b.bmodel       |    0.353 |    0.556 |
| SE9-16       | yolov8_opencv.py | yolov8s_int8_4b.bmodel       |    0.353 |    0.556 |
| SE9-16       | yolov8_bmcv.py | yolov8s_fp32_1b.bmodel         |    0.357 |    0.569 |
| SE9-16       | yolov8_bmcv.py | yolov8s_fp16_1b.bmodel         |    0.357 |    0.568 |
| SE9-16       | yolov8_bmcv.py | yolov8s_int8_1b.bmodel         |    0.353 |    0.556 |
| SE9-16       | yolov8_bmcv.py | yolov8s_int8_4b.bmodel         |    0.353 |    0.556 |
| SE9-16       | yolov8_bmcv.soc | yolov8s_fp32_1b.bmodel        |    0.351 |    0.567 |
| SE9-16       | yolov8_bmcv.soc | yolov8s_fp16_1b.bmodel        |    0.351 |    0.568 |
| SE9-16       | yolov8_bmcv.soc | yolov8s_int8_1b.bmodel        |    0.349 |    0.561 |
| SE9-16       | yolov8_bmcv.soc | yolov8s_int8_4b.bmodel        |    0.349 |    0.561 |
| SE9-16       | yolov8_opencv.py | yolov8s_fp32_1b_2core.bmodel |    0.358 |    0.569 |
| SE9-16       | yolov8_opencv.py | yolov8s_fp16_1b_2core.bmodel |    0.358 |    0.569 |
| SE9-16       | yolov8_opencv.py | yolov8s_int8_1b_2core.bmodel |    0.353 |    0.556 |
| SE9-16       | yolov8_opencv.py | yolov8s_int8_4b_2core.bmodel |    0.353 |    0.556 |
| SE9-16       | yolov8_bmcv.py | yolov8s_fp32_1b_2core.bmodel   |    0.357 |    0.569 |
| SE9-16       | yolov8_bmcv.py | yolov8s_fp16_1b_2core.bmodel   |    0.357 |    0.568 |
| SE9-16       | yolov8_bmcv.py | yolov8s_int8_1b_2core.bmodel   |    0.353 |    0.556 |
| SE9-16       | yolov8_bmcv.py | yolov8s_int8_4b_2core.bmodel   |    0.353 |    0.556 |
| SE9-16       | yolov8_bmcv.soc | yolov8s_fp32_1b_2core.bmodel  |    0.351 |    0.567 |
| SE9-16       | yolov8_bmcv.soc | yolov8s_fp16_1b_2core.bmodel  |    0.351 |    0.568 |
| SE9-16       | yolov8_bmcv.soc | yolov8s_int8_1b_2core.bmodel  |    0.349 |    0.561 |
| SE9-16       | yolov8_bmcv.soc | yolov8s_int8_4b_2core.bmodel  |    0.349 |    0.561 |

> **测试说明**：  
> 1. 由于sdk版本之间可能存在差异，实际运行结果与本表有<0.01的精度误差是正常的；
> 2. AP@IoU=0.5:0.95为area=all对应的指标；
> 3. 在搭载了相同TPU和SOPHONSDK的PCIe或SoC平台上，相同程序的精度一致，SE5系列对应BM1684，SE7系列对应BM1684X，SE9系列中，SE9-16对应BM1688；


## 7. 性能测试
### 7.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684/yolov8s_fp32_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

|              测试模型           | calculate time(ms) |
| -------------------------------| ----------------- |
| BM1684/yolov8s_fp32_1b.bmodel  |   38.5            |
| BM1684/yolov8s_int8_1b.bmodel  |   26.4            |
| BM1684/yolov8s_int8_4b.bmodel  |   16.3            |
| BM1684X/yolov8s_fp32_1b.bmodel |   39.7            |
| BM1684X/yolov8s_fp16_1b.bmodel |   7.5             |
| BM1684X/yolov8s_int8_1b.bmodel |   4.1             |
| BM1684X/yolov8s_int8_4b.bmodel |   3.7             |
| BM1688/yolov8s_fp32_1b.bmodel      |         233.50  |
| BM1688/yolov8s_fp16_1b.bmodel      |          45.65  |
| BM1688/yolov8s_int8_1b.bmodel      |          18.84  |
| BM1688/yolov8s_int8_4b.bmodel      |          17.87  |
| BM1688/yolov8s_fp32_1b_2core.bmodel|         122.36  |
| BM1688/yolov8s_fp16_1b_2core.bmodel|          26.44  |
| BM1688/yolov8s_int8_1b_2core.bmodel|          12.17  |
| BM1688/yolov8s_int8_4b_2core.bmodel|          10.32  |

> **测试说明**：  
1. 性能测试结果具有一定的波动性；
2. `calculate time`已折算为平均每张图片的推理时间；
3. SoC和PCIe的测试结果基本一致。


### 7.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++例程打印的预处理时间、推理时间、后处理时间为整个batch处理的时间，需除以相应的batch size才是每张图片的处理时间。

在不同的测试平台上，使用不同的例程、模型测试`datasets/val2017_1000`，conf_thresh=0.25，nms_thresh=0.7，性能测试结果如下：
|    测试平台  |     测试程序      |        测试模型        |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ---------------- | ---------------------- | -------- | --------- | --------- | --------- |
| SE5-16      | yolov8_opencv.py | yolov8s_fp32_1b.bmodel | 3.40 | 26.79 | 46.53 | 165.20 |
| SE5-16      | yolov8_opencv.py | yolov8s_int8_1b.bmodel | 3.39  | 27.03 | 34.33 | 162.30 | 
| SE5-16      | yolov8_opencv.py | yolov8s_int8_4b.bmodel | 4.05  | 25.76 | 24.19 | 139.20 |
| SE5-16      | yolov8_bmcv.py   | yolov8s_fp32_1b.bmodel | 2.84  | 2.99  | 43.67 | 181.50 |
| SE5-16      | yolov8_bmcv.py   | yolov8s_int8_1b.bmodel | 2.86  | 3.01  | 31.73 | 141.70 |
| SE5-16      | yolov8_bmcv.py   | yolov8s_int8_4b.bmodel | 2.51  | 2.82  | 21.48  | 135.00 |
| SE5-16      | yolov8_bmcv.soc  | yolov8s_fp32_1b.bmodel | 4.82  | 1.89  | 38.10 | 71.04 |
| SE5-16      | yolov8_bmcv.soc  | yolov8s_int8_1b.bmodel | 4.90  | 1.89  | 26.03 | 66.09 |
| SE5-16      | yolov8_bmcv.soc  | yolov8s_int8_4b.bmodel | 4.56  | 1.81  | 16.04 | 65.49 |
| SE7-32      | yolov8_opencv.py | yolov8s_fp32_1b.bmodel | 3.30   | 28.44 | 51.26 | 181.70 |
| SE7-32      | yolov8_opencv.py | yolov8s_fp16_1b.bmodel | 3.30   | 28.45 | 17.56 | 176.00 |
| SE7-32      | yolov8_opencv.py | yolov8s_int8_1b.bmodel | 3.33  | 27.99 | 13.74 | 161.50 |
| SE7-32      | yolov8_opencv.py | yolov8s_int8_4b.bmodel | 3.90   | 26.37 | 13.71 | 133.90 |
| SE7-32      | yolov8_bmcv.py   | yolov8s_fp32_1b.bmodel | 2.77  | 2.74  | 47.87 | 171.40 |
| SE7-32      | yolov8_bmcv.py   | yolov8s_fp16_1b.bmodel | 2.75  | 2.71  | 13.87 | 167.40 |
| SE7-32      | yolov8_bmcv.py   | yolov8s_int8_1b.bmodel | 2.73  | 2.70  | 10.32 | 153.50 |
| SE7-32      | yolov8_bmcv.py   | yolov8s_int8_4b.bmodel | 2.50  | 2.55  | 9.62  | 160.00 |
| SE7-32      | yolov8_bmcv.soc  | yolov8s_fp32_1b.bmodel | 4.66 | 0.99 | 41.55 | 70.40 |
| SE7-32      | yolov8_bmcv.soc  | yolov8s_fp16_1b.bmodel | 4.53 | 0.99 | 7.53 | 70.70 |
| SE7-32      | yolov8_bmcv.soc  | yolov8s_int8_1b.bmodel | 4.58 | 0.99 | 4.16 | 70.10 |
| SE7-32      | yolov8_bmcv.soc  | yolov8s_int8_4b.bmodel | 4.38 | 0.95 | 3.75 | 70.45 |
|   SE9-16    | yolov8_opencv.py  |      yolov8s_fp32_1b.bmodel       |      24.08      |      30.41      |     244.67      |     99.88       |
|   SE9-16    | yolov8_opencv.py  |      yolov8s_fp16_1b.bmodel       |      19.28      |      29.67      |      56.80      |     100.65      |
|   SE9-16    | yolov8_opencv.py  |      yolov8s_int8_1b.bmodel       |      11.62      |      30.25      |      30.19      |      92.74      |
|   SE9-16    | yolov8_opencv.py  |      yolov8s_int8_4b.bmodel       |      9.48       |      33.13      |      30.24      |      89.66      |
|   SE9-16    |  yolov8_bmcv.py   |      yolov8s_fp32_1b.bmodel       |      4.64       |      4.82       |     241.44      |     101.20      |
|   SE9-16    |  yolov8_bmcv.py   |      yolov8s_fp16_1b.bmodel       |      4.65       |      4.82       |      53.55      |     100.66      |
|   SE9-16    |  yolov8_bmcv.py   |      yolov8s_int8_1b.bmodel       |      4.71       |      4.85       |      26.72      |      90.80      |
|   SE9-16    |  yolov8_bmcv.py   |      yolov8s_int8_4b.bmodel       |      4.17       |      4.39       |      24.50      |      88.34      |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov8s_fp32_1b.bmodel       |      5.82       |      1.79       |     233.42      |     128.11      |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov8s_fp16_1b.bmodel       |      5.85       |      1.78       |      45.59      |     127.78      |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov8s_int8_1b.bmodel       |      5.82       |      1.77       |      18.79      |     116.44      |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov8s_int8_4b.bmodel       |      5.79       |      1.68       |      17.84      |     113.72      |
|   SE9-16    | yolov8_opencv.py  |   yolov8s_fp32_1b_2core.bmodel    |      19.26      |      30.54      |     133.66      |     100.09      |
|   SE9-16    | yolov8_opencv.py  |   yolov8s_fp16_1b_2core.bmodel    |      19.23      |      29.67      |      37.49      |     101.03      |
|   SE9-16    | yolov8_opencv.py  |   yolov8s_int8_1b_2core.bmodel    |      9.45       |      30.45      |      23.81      |      89.63      |
|   SE9-16    | yolov8_opencv.py  |   yolov8s_int8_4b_2core.bmodel    |      9.44       |      32.46      |      22.07      |      88.25      |
|   SE9-16    |  yolov8_bmcv.py   |   yolov8s_fp32_1b_2core.bmodel    |      4.71       |      4.82       |     130.19      |      98.58      |
|   SE9-16    |  yolov8_bmcv.py   |   yolov8s_fp16_1b_2core.bmodel    |      4.68       |      4.83       |      34.37      |      96.20      |
|   SE9-16    |  yolov8_bmcv.py   |   yolov8s_int8_1b_2core.bmodel    |      4.74       |      4.84       |      20.31      |      92.89      |
|   SE9-16    |  yolov8_bmcv.py   |   yolov8s_int8_4b_2core.bmodel    |      4.19       |      4.39       |      17.26      |      91.08      |
|   SE9-16    |  yolov8_bmcv.soc  |   yolov8s_fp32_1b_2core.bmodel    |      5.85       |      1.78       |     122.27      |     127.10      |
|   SE9-16    |  yolov8_bmcv.soc  |   yolov8s_fp16_1b_2core.bmodel    |      5.87       |      1.77       |      26.34      |     125.09      |
|   SE9-16    |  yolov8_bmcv.soc  |   yolov8s_int8_1b_2core.bmodel    |      5.84       |      1.77       |      12.11      |     114.18      |
|   SE9-16    |  yolov8_bmcv.soc  |   yolov8s_int8_4b_2core.bmodel    |      5.83       |      1.67       |      10.27      |     113.86      |

> **测试说明**：  
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. SE5-16/SE7-32的主控处理器均为8核CA53@2.3GHz，SE9-16为8核CA53@1.6GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异，不同的阈值对后处理时间影响较大。 


## 8. FAQ
请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。