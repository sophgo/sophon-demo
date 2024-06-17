# YOLOv9_seg

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
YOLOv9 引入了可编程梯度信息 (PGI) 和广义高效层聚合网络 (GELAN) 等开创性技术，标志着实时目标检测领域的重大进步。该模型在效率、准确性和适应性方面都有显著提高，在 MS COCO 数据集上树立了新的标杆。本例程对[​YOLOv9官方开源仓库](https://github.com/WongKinYiu/yolov9)的模型和算法进行移植，使之能在SOPHON BM1684/BM1684X/BM1688上进行推理测试。

## 2. 特性
* 支持BM1688(SoC)和BM1684X(x86 PCIe、SoC)和BM1684(x86 PCIe、SoC、arm PCIe)
* 支持FP32、FP16(BM1684X/BM1688)、INT8模型编译和推理
* 支持基于BMCV预处理的C++推理
* 支持基于OpenCV和BMCV预处理的Python推理
* 支持单batch和多batch模型推理
* 支持1个输出模型推理
* 支持图片和视频测试

## 3. 准备模型与数据
建议使用TPU-MLIR编译BModel，在使用TPU-MLIR编译前需要导出ONNX模型。具体可参考[YOLOv9模型导出](./docs/YOLOv9_Export_Guide.md)。

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
│   ├── yolov9c_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684的FP32 BModel，batch_size=1
│   ├── yolov9c_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684的INT8 BModel，batch_size=1
│   └── yolov9c_int8_4b.bmodel   # 使用TPU-MLIR编译，用于BM1684的INT8 BModel，batch_size=4
├── BM1684X
│   ├── yolov9c_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=1
│   ├── yolov9c_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=1
│   ├── yolov9c_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=1
│   └── yolov9c_int8_4b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=4
├── BM1688
│   ├── yolov9c_fp32_1b.bmodel        # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1，num_core=1
│   ├── yolov9c_fp16_1b.bmodel        # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1，num_core=1
│   ├── yolov9c_int8_1b.bmodel        # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=1，num_core=1
│   ├── yolov9c_int8_4b.bmodel        # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=4，num_core=1
│   ├── yolov9c_fp32_1b_2core.bmodel  # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1，num_core=2
│   ├── yolov9c_fp16_1b_2core.bmodel  # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1，num_core=2
│   ├── yolov9c_int8_1b_2core.bmodel  # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=1，num_core=2
│   └── yolov9c_int8_4b_2core.bmodel  # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=4，num_core=2
└── onnx
    ├── yolov9c_1b.onnx           # 导出的静态onnx模型，batch_size=1
    ├── yolov9c_4b.onnx           # 导出的静态onnx模型，batch_size=4
    ├── yolov9c_bm1684_qtable     # TPU-MLIR编译时，用于BM1684的INT8 BModel混合精度量化
    ├── yolov9c_bm1684x_qtable    # TPU-MLIR编译时，用于BM1684X的INT8 BModel混合精度量化
    └── yolov9c_bm1688_qtable     # TPU-MLIR编译时，用于BM1688的INT8 BModel混合精度量化
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

模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index/material/31/all.html)相应版本的SDK中获取)。

- 生成FP32 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684/BM1684X/BM1688**），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684 #bm1688
```

​执行上述命令会在`models/BM1684`下生成`yolov9c_fp32_1b.bmodel`文件，即转换好的FP32 BModel。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688**），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x #bm1688
```

​执行上述命令会在`models/BM1684X/`下生成`yolov9c_fp16_1b.bmodel`文件，即转换好的FP16 BModel。

- 生成INT8 BModel

​本例程在`scripts`目录下提供了量化INT8 BModel的脚本，请注意修改`gen_int8bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台（**支持BM1684/BM1684X/BM1688**），如：

```bash
./scripts/gen_int8bmodel_mlir.sh bm1684 #bm1684x/bm1688

```

​执行上述命令会在`models/BM1684`下生成`yolov9c_int8_1b.bmodel`等文件，即转换好的INT8 BModel。

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
python3 tools/eval_coco.py --gt_path datasets/coco/instances_val2017_1000.json --result_json results/yolov9c_fp32_1b.bmodel_val2017_1000_opencv_python_result.json
```
### 6.2 测试结果
在coco2017 val数据集上，精度测试结果如下：
|   测试平台    |      测试程序     |      测试模型          |AP@IoU=0.5:0.95|AP@IoU=0.5|
| ------------ | ---------------- | ---------------------- | ------------- | -------- |
| SE5-16       | yolov9_opencv.py | yolov9c_fp32_1b.bmodel |    0.417 |    0.644 |
| SE5-16       | yolov9_opencv.py | yolov9c_int8_1b.bmodel |    0.401 |    0.623 |
| SE5-16       | yolov9_opencv.py | yolov9c_int8_4b.bmodel |    0.401 |    0.623 |
| SE5-16       | yolov9_bmcv.py | yolov9c_fp32_1b.bmodel |    0.417 |    0.644 |
| SE5-16       | yolov9_bmcv.py | yolov9c_int8_1b.bmodel |    0.400 |    0.623 |
| SE5-16       | yolov9_bmcv.py | yolov9c_int8_4b.bmodel |    0.400 |    0.623 |
| SE5-16       | yolov9_bmcv.soc | yolov9c_fp32_1b.bmodel |    0.404 |    0.641 |
| SE5-16       | yolov9_bmcv.soc | yolov9c_int8_1b.bmodel |    0.389 |    0.620 |
| SE5-16       | yolov9_bmcv.soc | yolov9c_int8_4b.bmodel |    0.389 |    0.620 |
| SE7-32       | yolov9_opencv.py | yolov9c_fp32_1b.bmodel |    0.417 |    0.644 |
| SE7-32       | yolov9_opencv.py | yolov9c_fp16_1b.bmodel |    0.417 |    0.644 |
| SE7-32       | yolov9_opencv.py | yolov9c_int8_1b.bmodel |    0.416 |    0.639 |
| SE7-32       | yolov9_opencv.py | yolov9c_int8_4b.bmodel |    0.416 |    0.639 |
| SE7-32       | yolov9_bmcv.py | yolov9c_fp32_1b.bmodel |    0.416 |    0.644 |
| SE7-32       | yolov9_bmcv.py | yolov9c_fp16_1b.bmodel |    0.416 |    0.644 |
| SE7-32       | yolov9_bmcv.py | yolov9c_int8_1b.bmodel |    0.416 |    0.638 |
| SE7-32       | yolov9_bmcv.py | yolov9c_int8_4b.bmodel |    0.416 |    0.638 |
| SE7-32       | yolov9_bmcv.soc | yolov9c_fp32_1b.bmodel |    0.404 |    0.641 |
| SE7-32       | yolov9_bmcv.soc | yolov9c_fp16_1b.bmodel |    0.405 |    0.641 |
| SE7-32       | yolov9_bmcv.soc | yolov9c_int8_1b.bmodel |    0.404 |    0.638 |
| SE7-32       | yolov9_bmcv.soc | yolov9c_int8_4b.bmodel |    0.404 |    0.638 |
| SE9-16       | yolov9_opencv.py | yolov9c_fp32_1b.bmodel |    0.417 |    0.644 |
| SE9-16       | yolov9_opencv.py | yolov9c_fp16_1b.bmodel |    0.417 |    0.645 |
| SE9-16       | yolov9_opencv.py | yolov9c_int8_1b.bmodel |    0.416 |    0.639 |
| SE9-16       | yolov9_opencv.py | yolov9c_int8_4b.bmodel |    0.416 |    0.639 |
| SE9-16       | yolov9_bmcv.py | yolov9c_fp32_1b.bmodel |    0.416 |    0.644 |
| SE9-16       | yolov9_bmcv.py | yolov9c_fp16_1b.bmodel |    0.416 |    0.644 |
| SE9-16       | yolov9_bmcv.py | yolov9c_int8_1b.bmodel |    0.414 |    0.638 |
| SE9-16       | yolov9_bmcv.py | yolov9c_int8_4b.bmodel |    0.414 |    0.638 |
| SE9-16       | yolov9_bmcv.soc | yolov9c_fp32_1b.bmodel |    0.404 |    0.641 |
| SE9-16       | yolov9_bmcv.soc | yolov9c_fp16_1b.bmodel |    0.405 |    0.641 |
| SE9-16       | yolov9_bmcv.soc | yolov9c_int8_1b.bmodel |    0.405 |    0.638 |
| SE9-16       | yolov9_bmcv.soc | yolov9c_int8_4b.bmodel |    0.405 |    0.638 |
| SE9-16       | yolov9_opencv.py | yolov9c_fp32_1b_2core.bmodel |    0.417 |    0.644 |
| SE9-16       | yolov9_opencv.py | yolov9c_fp16_1b_2core.bmodel |    0.417 |    0.645 |
| SE9-16       | yolov9_opencv.py | yolov9c_int8_1b_2core.bmodel |    0.416 |    0.639 |
| SE9-16       | yolov9_opencv.py | yolov9c_int8_4b_2core.bmodel |    0.416 |    0.639 |
| SE9-16       | yolov9_bmcv.py | yolov9c_fp32_1b_2core.bmodel |    0.416 |    0.644 |
| SE9-16       | yolov9_bmcv.py | yolov9c_fp16_1b_2core.bmodel |    0.416 |    0.644 |
| SE9-16       | yolov9_bmcv.py | yolov9c_int8_1b_2core.bmodel |    0.414 |    0.638 |
| SE9-16       | yolov9_bmcv.py | yolov9c_int8_4b_2core.bmodel |    0.414 |    0.638 |
| SE9-16       | yolov9_bmcv.soc | yolov9c_fp32_1b_2core.bmodel |    0.404 |    0.641 |
| SE9-16       | yolov9_bmcv.soc | yolov9c_fp16_1b_2core.bmodel |    0.405 |    0.641 |
| SE9-16       | yolov9_bmcv.soc | yolov9c_int8_1b_2core.bmodel |    0.405 |    0.638 |
| SE9-16       | yolov9_bmcv.soc | yolov9c_int8_4b_2core.bmodel |    0.405 |    0.638 |

> **测试说明**：
> 1. 由于sdk版本之间可能存在差异，实际运行结果与本表有<0.01的精度误差是正常的；
> 2. AP@IoU=0.5:0.95为area=all对应的指标；
> 3. 在搭载了相同TPU和SOPHONSDK的PCIe或SoC平台上，相同程序的精度一致，SE5系列对应BM1684，SE7系列对应BM1684X，SE9系列中，SE9-16对应BM1688；


## 7. 性能测试
### 7.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684/yolov9c_fp32_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

|              测试模型           | calculate time(ms) |
| -------------------------------| ----------------- |
| BM1684/yolov9c_fp32_1b.bmodel      |         108.30  |
| BM1684/yolov9c_int8_1b.bmodel      |          57.20  |
| BM1684/yolov9c_int8_4b.bmodel      |          24.53  |
| BM1684X/yolov9c_fp32_1b.bmodel     |         137.45  |
| BM1684X/yolov9c_fp16_1b.bmodel     |          21.44  |
| BM1684X/yolov9c_int8_1b.bmodel     |           9.86  |
| BM1684X/yolov9c_int8_4b.bmodel     |           9.32  |
| BM1688/yolov9c_fp32_1b.bmodel      |         781.29  |
| BM1688/yolov9c_fp16_1b.bmodel      |         146.76  |
| BM1688/yolov9c_int8_1b.bmodel      |          31.21  |
| BM1688/yolov9c_int8_4b.bmodel      |          30.76  |
| BM1688/yolov9c_fp32_1b_2core.bmodel|         404.24  |
| BM1688/yolov9c_fp16_1b_2core.bmodel|          80.10  |
| BM1688/yolov9c_int8_1b_2core.bmodel|          22.22  |
| BM1688/yolov9c_int8_4b_2core.bmodel|          17.92  |

> **测试说明**：
1. 性能测试结果具有一定的波动性；
2. `calculate time`已折算为平均每张图片的推理时间；
3. SoC和PCIe的测试结果基本一致。


### 7.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++例程打印的预处理时间、推理时间、后处理时间为整个batch处理的时间，需除以相应的batch size才是每张图片的处理时间。

在不同的测试平台上，使用不同的例程、模型测试`datasets/coco/val2017_1000`，conf_thresh=0.25，nms_thresh=0.7，性能测试结果如下：
|    测试平台  |     测试程序      |        测试模型        |decode_time|preprocess_time|inference_time|postprocess_time|
| ----------- | ---------------- | ---------------------- | -------- | --------- | --------- | --------- |
|   SE5-16    | yolov9_opencv.py  |      yolov9c_fp32_1b.bmodel       |      6.82       |      21.83      |     116.21      |      72.16      |
|   SE5-16    | yolov9_opencv.py  |      yolov9c_int8_1b.bmodel       |      6.84       |      21.95      |      65.04      |      68.01      |
|   SE5-16    | yolov9_opencv.py  |      yolov9c_int8_4b.bmodel       |      6.86       |      23.20      |      32.77      |      64.29      |
|   SE5-16    |  yolov9_bmcv.py   |      yolov9c_fp32_1b.bmodel       |      3.84       |      2.80       |     113.79      |      77.57      |
|   SE5-16    |  yolov9_bmcv.py   |      yolov9c_int8_1b.bmodel       |      3.88       |      2.81       |      62.65      |      71.35      |
|   SE5-16    |  yolov9_bmcv.py   |      yolov9c_int8_4b.bmodel       |      3.53       |      2.59       |      29.32      |      68.15      |
|   SE5-16    |  yolov9_bmcv.soc  |      yolov9c_fp32_1b.bmodel       |      5.02       |      1.55       |     108.19      |      83.33      |
|   SE5-16    |  yolov9_bmcv.soc  |      yolov9c_int8_1b.bmodel       |      5.03       |      1.55       |      57.07      |      76.70      |
|   SE5-16    |  yolov9_bmcv.soc  |      yolov9c_int8_4b.bmodel       |      4.99       |      1.49       |      24.48      |      73.58      |
|   SE7-32    | yolov9_opencv.py  |      yolov9c_fp32_1b.bmodel       |      6.88       |      23.32      |     146.29      |      79.88      |
|   SE7-32    | yolov9_opencv.py  |      yolov9c_fp16_1b.bmodel       |      6.83       |      22.81      |      30.36      |      80.03      |
|   SE7-32    | yolov9_opencv.py  |      yolov9c_int8_1b.bmodel       |      6.79       |      22.75      |      18.81      |      73.48      |
|   SE7-32    | yolov9_opencv.py  |      yolov9c_int8_4b.bmodel       |      6.83       |      22.59      |      17.23      |      73.51      |
|   SE7-32    |  yolov9_bmcv.py   |      yolov9c_fp32_1b.bmodel       |      3.35       |      2.35       |     143.56      |      82.37      |
|   SE7-32    |  yolov9_bmcv.py   |      yolov9c_fp16_1b.bmodel       |      3.35       |      2.35       |      27.55      |      83.45      |
|   SE7-32    |  yolov9_bmcv.py   |      yolov9c_int8_1b.bmodel       |      3.35       |      2.36       |      15.95      |      76.93      |
|   SE7-32    |  yolov9_bmcv.py   |      yolov9c_int8_4b.bmodel       |      3.00       |      2.13       |      14.72      |      77.48      |
|   SE7-32    |  yolov9_bmcv.soc  |      yolov9c_fp32_1b.bmodel       |      4.45       |      0.74       |     137.37      |      89.15      |
|   SE7-32    |  yolov9_bmcv.soc  |      yolov9c_fp16_1b.bmodel       |      4.49       |      0.74       |      21.39      |      91.95      |
|   SE7-32    |  yolov9_bmcv.soc  |      yolov9c_int8_1b.bmodel       |      4.52       |      0.74       |      9.79       |      82.50      |
|   SE7-32    |  yolov9_bmcv.soc  |      yolov9c_int8_4b.bmodel       |      4.50       |      0.71       |      9.30       |      82.11      |
|   SE9-16    | yolov9_opencv.py  |      yolov9c_fp32_1b.bmodel       |      9.48       |      29.76      |     792.35      |     101.83      |
|   SE9-16    | yolov9_opencv.py  |      yolov9c_fp16_1b.bmodel       |      9.47       |      29.93      |     158.00      |     105.56      |
|   SE9-16    | yolov9_opencv.py  |      yolov9c_int8_1b.bmodel       |      9.51       |      30.48      |      42.30      |      92.94      |
|   SE9-16    | yolov9_opencv.py  |      yolov9c_int8_4b.bmodel       |      9.45       |      29.78      |      40.45      |      92.51      |
|   SE9-16    |  yolov9_bmcv.py   |      yolov9c_fp32_1b.bmodel       |      4.65       |      4.71       |     789.43      |     105.47      |
|   SE9-16    |  yolov9_bmcv.py   |      yolov9c_fp16_1b.bmodel       |      4.67       |      4.71       |     154.50      |     101.52      |
|   SE9-16    |  yolov9_bmcv.py   |      yolov9c_int8_1b.bmodel       |      4.77       |      4.69       |      38.99      |      96.56      |
|   SE9-16    |  yolov9_bmcv.py   |      yolov9c_int8_4b.bmodel       |      4.21       |      4.28       |      37.62      |      95.59      |
|   SE9-16    |  yolov9_bmcv.soc  |      yolov9c_fp32_1b.bmodel       |      5.87       |      1.74       |     781.26      |     119.35      |
|   SE9-16    |  yolov9_bmcv.soc  |      yolov9c_fp16_1b.bmodel       |      5.88       |      1.74       |     146.66      |     118.59      |
|   SE9-16    |  yolov9_bmcv.soc  |      yolov9c_int8_1b.bmodel       |      5.88       |      1.73       |      31.11      |     110.35      |
|   SE9-16    |  yolov9_bmcv.soc  |      yolov9c_int8_4b.bmodel       |      5.87       |      1.66       |      30.72      |     110.37      |
|   SE9-16    | yolov9_opencv.py  |   yolov9c_fp32_1b_2core.bmodel    |      9.49       |      30.45      |     415.48      |     100.86      |
|   SE9-16    | yolov9_opencv.py  |   yolov9c_fp16_1b_2core.bmodel    |      9.48       |      30.51      |      91.45      |     100.04      |
|   SE9-16    | yolov9_opencv.py  |   yolov9c_int8_1b_2core.bmodel    |      9.45       |      30.56      |      33.31      |      94.72      |
|   SE9-16    | yolov9_opencv.py  |   yolov9c_int8_4b_2core.bmodel    |      9.48       |      32.91      |      29.48      |      90.92      |
|   SE9-16    |  yolov9_bmcv.py   |   yolov9c_fp32_1b_2core.bmodel    |      4.69       |      4.72       |     412.10      |     106.78      |
|   SE9-16    |  yolov9_bmcv.py   |   yolov9c_fp16_1b_2core.bmodel    |      4.69       |      4.77       |      87.89      |     100.90      |
|   SE9-16    |  yolov9_bmcv.py   |   yolov9c_int8_1b_2core.bmodel    |      4.68       |      4.75       |      30.05      |      96.75      |
|   SE9-16    |  yolov9_bmcv.py   |   yolov9c_int8_4b_2core.bmodel    |      4.25       |      4.30       |      24.49      |      96.67      |
|   SE9-16    |  yolov9_bmcv.soc  |   yolov9c_fp32_1b_2core.bmodel    |      5.91       |      1.74       |     404.17      |     118.55      |
|   SE9-16    |  yolov9_bmcv.soc  |   yolov9c_fp16_1b_2core.bmodel    |      5.90       |      1.73       |      80.05      |     118.54      |
|   SE9-16    |  yolov9_bmcv.soc  |   yolov9c_int8_1b_2core.bmodel    |      5.92       |      1.74       |      22.14      |     110.50      |
|   SE9-16    |  yolov9_bmcv.soc  |   yolov9c_int8_4b_2core.bmodel    |      5.87       |      1.65       |      17.90      |     110.19      |

> **测试说明**：
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. SE5-16/SE7-32的主控处理器均为8核CA53@2.3GHz，SE9-16为8核CA53@1.6GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异，不同的阈值对后处理时间影响较大。


## 8. FAQ
请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。
