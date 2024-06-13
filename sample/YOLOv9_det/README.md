# YOLOv9

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
YOLOv9 引入了可编程梯度信息 (PGI) 和广义高效层聚合网络 (GELAN) 等开创性技术，标志着实时目标检测领域的重大进步。该模型在效率、准确性和适应性方面都有显著提高，在 MS COCO 数据集上树立了新的标杆。本例程对[​YOLOv9官方开源仓库](https://github.com/WongKinYiu/yolov9)的模型和算法进行移植，使之能在SOPHON BM1684/BM1684X/BM1688/CV186X上进行推理测试。

## 2. 特性
* 支持BM1688/CV186X(SoC)、BM1684X(x86 PCIe、SoC)、BM1684(x86 PCIe、SoC、arm PCIe)
* 支持FP32、FP16(BM1684X/BM1688/CV186X)、INT8模型编译和推理
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
│   ├── yolov9s_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684的FP32 BModel，batch_size=1
│   ├── yolov9s_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684的INT8 BModel，batch_size=1
│   └── yolov9s_int8_4b.bmodel   # 使用TPU-MLIR编译，用于BM1684的INT8 BModel，batch_size=4
├── BM1684X
│   ├── yolov9s_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=1
│   ├── yolov9s_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=1
│   ├── yolov9s_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=1
│   └── yolov9s_int8_4b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=4
├── BM1688
│   ├── yolov9s_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1, num_core=1
│   ├── yolov9s_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1, num_core=1
│   ├── yolov9s_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=1, num_core=1
│   ├── yolov9s_int8_4b.bmodel   # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=4, num_core=1
│   ├── yolov9s_fp32_1b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1, num_core=2
│   ├── yolov9s_fp16_1b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1, num_core=2
│   ├── yolov9s_int8_1b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=1, num_core=2
│   └── yolov9s_int8_4b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=4, num_core=2
├── CV186X
│   ├── yolov9s_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于CV186X的FP32 BModel，batch_size=1
│   ├── yolov9s_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于CV186X的FP16 BModel，batch_size=1
│   ├── yolov9s_int8_1b.bmodel   # 使用TPU-MLIR编译，用于CV186X的INT8 BModel，batch_size=1
│   └── yolov9s_int8_4b.bmodel   # 使用TPU-MLIR编译，用于CV186X的INT8 BModel，batch_size=4
└── onnx
    ├── yolov9s_1b.onnx      # 导出的动态onnx模型，batch_size=1
    ├── yolov9s_4b.onnx      # 导出的动态onnx模型，batch_size=4
    ├── yolov9s_qtable_fp16       # TPU-MLIR编译时，用于BM1684X/BM1688的INT8 BModel混合精度量化
    ├── yolov9s_qtable_fp32       # TPU-MLIR编译时，用于BM1684的INT8 BModel混合精度量化
    └── yolov9s_qtable_bf16       # TPU-MLIR编译时，用于CV186X的INT8 BModel混合精度量化



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

模型编译前需要安装TPU-MLIR，本例程使用的TPU-MLIR版本是`v1.6`，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index/material/31/all.html)相应版本的SDK中获取)。

- 生成FP32 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684/BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684 #bm1684x/bm1688/cv186x
```

​执行上述命令会在`models/BM1684`下生成`yolov9s_fp32_1b.bmodel`文件，即转换好的FP32 BModel。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x #bm1688/cv186x
```

​执行上述命令会在`models/BM1684X/`下生成`yolov9s_fp16_1b.bmodel`文件，即转换好的FP16 BModel。

- 生成INT8 BModel

​本例程在`scripts`目录下提供了量化INT8 BModel的脚本，请注意修改`gen_int8bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台（**支持BM1684/BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_int8bmodel_mlir.sh bm1684 #bm1684x/bm1688/cv186x
```

​执行上述命令会在`models/BM1684`下生成`yolov9s_int8_1b.bmodel`等文件，即转换好的INT8 BModel。量化模型出现问题可以参考：[Calibration_Guide](../../docs/Calibration_Guide.md)。


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
python3 tools/eval_coco.py --gt_path datasets/coco/instances_val2017_1000.json --result_json results/yolov9s_fp32_1b.bmodel_val2017_1000_opencv_python_result.json
```
### 6.2 测试结果
在coco2017 val数据集上，精度测试结果如下：
|   测试平台    |      测试程序     |      测试模型          |AP@IoU=0.5:0.95|AP@IoU=0.5|
| ------------ | ---------------- | ---------------------- | ------------- | -------- |
| SE5-16       | yolov9_opencv.py | yolov9s_fp32_1b.bmodel |    0.465 |    0.630 |
| SE5-16       | yolov9_opencv.py | yolov9s_int8_1b.bmodel |    0.436 |    0.597 |
| SE5-16       | yolov9_opencv.py | yolov9s_int8_4b.bmodel |    0.436 |    0.597 |
| SE5-16       | yolov9_bmcv.py | yolov9s_fp32_1b.bmodel |    0.464 |    0.630 |
| SE5-16       | yolov9_bmcv.py | yolov9s_int8_1b.bmodel |    0.435 |    0.596 |
| SE5-16       | yolov9_bmcv.py | yolov9s_int8_4b.bmodel |    0.435 |    0.596 |
| SE5-16       | yolov9_bmcv.soc | yolov9s_fp32_1b.bmodel |    0.465 |    0.630 |
| SE5-16       | yolov9_bmcv.soc | yolov9s_int8_1b.bmodel |    0.436 |    0.597 |
| SE5-16       | yolov9_bmcv.soc | yolov9s_int8_4b.bmodel |    0.436 |    0.597 |
| SE7-32       | yolov9_opencv.py | yolov9s_fp32_1b.bmodel |    0.465 |    0.630 |
| SE7-32       | yolov9_opencv.py | yolov9s_fp16_1b.bmodel |    0.464 |    0.630 |
| SE7-32       | yolov9_opencv.py | yolov9s_int8_1b.bmodel |    0.461 |    0.625 |
| SE7-32       | yolov9_opencv.py | yolov9s_int8_4b.bmodel |    0.461 |    0.625 |
| SE7-32       | yolov9_bmcv.py | yolov9s_fp32_1b.bmodel |    0.464 |    0.630 |
| SE7-32       | yolov9_bmcv.py | yolov9s_fp16_1b.bmodel |    0.463 |    0.630 |
| SE7-32       | yolov9_bmcv.py | yolov9s_int8_1b.bmodel |    0.461 |    0.626 |
| SE7-32       | yolov9_bmcv.py | yolov9s_int8_4b.bmodel |    0.461 |    0.626 |
| SE7-32       | yolov9_bmcv.soc | yolov9s_fp32_1b.bmodel |    0.465 |    0.630 |
| SE7-32       | yolov9_bmcv.soc | yolov9s_fp16_1b.bmodel |    0.464 |    0.629 |
| SE7-32       | yolov9_bmcv.soc | yolov9s_int8_1b.bmodel |    0.461 |    0.625 |
| SE7-32       | yolov9_bmcv.soc | yolov9s_int8_4b.bmodel |    0.461 |    0.625 |
| SE9-16       | yolov9_opencv.py | yolov9s_fp32_1b.bmodel |    0.465 |    0.630 |
| SE9-16       | yolov9_opencv.py | yolov9s_fp16_1b.bmodel |    0.464 |    0.630 |
| SE9-16       | yolov9_opencv.py | yolov9s_int8_1b.bmodel |    0.461 |    0.625 |
| SE9-16       | yolov9_opencv.py | yolov9s_int8_4b.bmodel |    0.461 |    0.625 |
| SE9-16       | yolov9_bmcv.py | yolov9s_fp32_1b.bmodel |    0.465 |    0.630 |
| SE9-16       | yolov9_bmcv.py | yolov9s_fp16_1b.bmodel |    0.463 |    0.630 |
| SE9-16       | yolov9_bmcv.py | yolov9s_int8_1b.bmodel |    0.461 |    0.627 |
| SE9-16       | yolov9_bmcv.py | yolov9s_int8_4b.bmodel |    0.461 |    0.627 |
| SE9-16       | yolov9_bmcv.soc | yolov9s_fp32_1b.bmodel |    0.465 |    0.630 |
| SE9-16       | yolov9_bmcv.soc | yolov9s_fp16_1b.bmodel |    0.464 |    0.630 |
| SE9-16       | yolov9_bmcv.soc | yolov9s_int8_1b.bmodel |    0.462 |    0.626 |
| SE9-16       | yolov9_bmcv.soc | yolov9s_int8_4b.bmodel |    0.462 |    0.626 |
| SE9-16       | yolov9_opencv.py | yolov9s_fp32_1b_2core.bmodel |    0.465 |    0.630 |
| SE9-16       | yolov9_opencv.py | yolov9s_fp16_1b_2core.bmodel |    0.464 |    0.630 |
| SE9-16       | yolov9_opencv.py | yolov9s_int8_1b_2core.bmodel |    0.461 |    0.625 |
| SE9-16       | yolov9_opencv.py | yolov9s_int8_4b_2core.bmodel |    0.461 |    0.625 |
| SE9-16       | yolov9_bmcv.py | yolov9s_fp32_1b_2core.bmodel |    0.465 |    0.630 |
| SE9-16       | yolov9_bmcv.py | yolov9s_fp16_1b_2core.bmodel |    0.463 |    0.630 |
| SE9-16       | yolov9_bmcv.py | yolov9s_int8_1b_2core.bmodel |    0.461 |    0.627 |
| SE9-16       | yolov9_bmcv.py | yolov9s_int8_4b_2core.bmodel |    0.461 |    0.627 |
| SE9-16       | yolov9_bmcv.soc | yolov9s_fp32_1b_2core.bmodel |    0.465 |    0.630 |
| SE9-16       | yolov9_bmcv.soc | yolov9s_fp16_1b_2core.bmodel |    0.464 |    0.630 |
| SE9-16       | yolov9_bmcv.soc | yolov9s_int8_1b_2core.bmodel |    0.462 |    0.626 |
| SE9-16       | yolov9_bmcv.soc | yolov9s_int8_4b_2core.bmodel |    0.462 |    0.626 |
| SE9-8        | yolov9_opencv.py | yolov9s_fp32_1b.bmodel |    0.465 |    0.630 |
| SE9-8        | yolov9_opencv.py | yolov9s_fp16_1b.bmodel |    0.464 |    0.630 |
| SE9-8        | yolov9_opencv.py | yolov9s_int8_1b.bmodel |    0.461 |    0.625 |
| SE9-8        | yolov9_opencv.py | yolov9s_int8_4b.bmodel |    0.461 |    0.625 |
| SE9-8        | yolov9_bmcv.py | yolov9s_fp32_1b.bmodel |    0.465 |    0.630 |
| SE9-8        | yolov9_bmcv.py | yolov9s_fp16_1b.bmodel |    0.463 |    0.630 |
| SE9-8        | yolov9_bmcv.py | yolov9s_int8_1b.bmodel |    0.461 |    0.627 |
| SE9-8        | yolov9_bmcv.py | yolov9s_int8_4b.bmodel |    0.461 |    0.627 |
| SE9-8        | yolov9_bmcv.soc | yolov9s_fp32_1b.bmodel |    0.465 |    0.630 |
| SE9-8        | yolov9_bmcv.soc | yolov9s_fp16_1b.bmodel |    0.464 |    0.630 |
| SE9-8        | yolov9_bmcv.soc | yolov9s_int8_1b.bmodel |    0.462 |    0.626 |
| SE9-8        | yolov9_bmcv.soc | yolov9s_int8_4b.bmodel |    0.462 |    0.626 |
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
bmrt_test --bmodel models/BM1684/yolov9s_fp32_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

|              测试模型               | calculate time(ms) |
| ----------------------------------- | ----------------- |
| BM1684/yolov9s_fp32_1b.bmodel      |          36.52  |
| BM1684/yolov9s_int8_1b.bmodel      |          25.69  |
| BM1684/yolov9s_int8_4b.bmodel      |          18.66  |
| BM1684X/yolov9s_fp32_1b.bmodel     |          33.77  |
| BM1684X/yolov9s_fp16_1b.bmodel     |           7.50  |
| BM1684X/yolov9s_int8_1b.bmodel     |           5.13  |
| BM1684X/yolov9s_int8_4b.bmodel     |           4.61  |
| BM1688/yolov9s_fp32_1b.bmodel      |         162.39  |
| BM1688/yolov9s_fp16_1b.bmodel      |          40.97  |
| BM1688/yolov9s_int8_1b.bmodel      |          18.25  |
| BM1688/yolov9s_int8_4b.bmodel      |          17.73  |
| BM1688/yolov9s_fp32_1b_2core.bmodel|          91.24  |
| BM1688/yolov9s_fp16_1b_2core.bmodel|          24.95  |
| BM1688/yolov9s_int8_1b_2core.bmodel|          12.71  |
| BM1688/yolov9s_int8_4b_2core.bmodel|          10.13  |
| CV186X/yolov9s_fp32_1b.bmodel      |         162.44  |
| CV186X/yolov9s_fp16_1b.bmodel      |          41.03  |
| CV186X/yolov9s_int8_1b.bmodel      |          18.28  |
| CV186X/yolov9s_int8_4b.bmodel      |          17.68  |
> **测试说明**：
1. 性能测试结果具有一定的波动性；
2. `calculate time`已折算为平均每张图片的推理时间；
3. SoC和PCIe的测试结果基本一致。


### 7.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++和Python例程打印的时间已经折算为单张图片的处理时间。

在不同的测试平台上，使用不同的例程、模型测试`datasets/coco/val2017_1000`，conf_thresh=0.25，nms_thresh=0.7，性能测试结果如下：
|    测试平台  |     测试程序      |        测试模型        |decode_time|preprocess_time|inference_time|postprocess_time|
| ----------- | ---------------- | ---------------------- | --------  | ---------    | ---------     | ---------      |
|   SE5-16    | yolov9_opencv.py  | yolov9s_fp32_1b.bmodel  |      6.81       |      21.85      |      41.59      |      5.07       |
|   SE5-16    | yolov9_opencv.py  | yolov9s_int8_1b.bmodel  |      6.83       |      22.08      |      32.84      |      4.95       |
|   SE5-16    | yolov9_opencv.py  | yolov9s_int8_4b.bmodel  |      6.90       |      24.67      |      29.09      |      5.16       |
|   SE5-16    |  yolov9_bmcv.py   | yolov9s_fp32_1b.bmodel  |      3.59       |      2.75       |      38.77      |      4.97       |
|   SE5-16    |  yolov9_bmcv.py   | yolov9s_int8_1b.bmodel  |      3.74       |      2.89       |      28.09      |      4.93       |
|   SE5-16    |  yolov9_bmcv.py   | yolov9s_int8_4b.bmodel  |      3.45       |      2.56       |      20.58      |      4.35       |
|   SE5-16    |  yolov9_bmcv.soc  | yolov9s_fp32_1b.bmodel  |      4.87       |      1.55       |      36.39      |      8.53       |
|   SE5-16    |  yolov9_bmcv.soc  | yolov9s_int8_1b.bmodel  |      4.89       |      1.55       |      25.58      |      8.53       |
|   SE5-16    |  yolov9_bmcv.soc  | yolov9s_int8_4b.bmodel  |      4.76       |      1.49       |      18.62      |      8.49       |
|   SE7-32    | yolov9_opencv.py  | yolov9s_fp32_1b.bmodel  |      6.74       |      22.39      |      39.55      |      5.40       |
|   SE7-32    | yolov9_opencv.py  | yolov9s_fp16_1b.bmodel  |      6.80       |      22.85      |      13.27      |      5.38       |
|   SE7-32    | yolov9_opencv.py  | yolov9s_int8_1b.bmodel  |      6.80       |      22.79      |      10.79      |      5.33       |
|   SE7-32    | yolov9_opencv.py  | yolov9s_int8_4b.bmodel  |      6.83       |      24.81      |      10.18      |      5.41       |
|   SE7-32    |  yolov9_bmcv.py   | yolov9s_fp32_1b.bmodel  |      3.12       |      2.30       |      36.29      |      5.42       |
|   SE7-32    |  yolov9_bmcv.py   | yolov9s_fp16_1b.bmodel  |      3.10       |      2.30       |      10.03      |      5.46       |
|   SE7-32    |  yolov9_bmcv.py   | yolov9s_int8_1b.bmodel  |      3.10       |      2.28       |      7.58       |      5.42       |
|   SE7-32    |  yolov9_bmcv.py   | yolov9s_int8_4b.bmodel  |      2.94       |      2.11       |      6.76       |      4.88       |
|   SE7-32    |  yolov9_bmcv.soc  | yolov9s_fp32_1b.bmodel  |      4.31       |      0.74       |      33.67      |      8.64       |
|   SE7-32    |  yolov9_bmcv.soc  | yolov9s_fp16_1b.bmodel  |      4.33       |      0.74       |      7.41       |      8.65       |
|   SE7-32    |  yolov9_bmcv.soc  | yolov9s_int8_1b.bmodel  |      4.36       |      0.74       |      4.95       |      8.64       |
|   SE7-32    |  yolov9_bmcv.soc  | yolov9s_int8_4b.bmodel  |      4.19       |      0.71       |      4.57       |      8.59       |
|   SE9-16    | yolov9_opencv.py  | yolov9s_fp32_1b.bmodel  |      9.53       |      29.52      |     169.66      |      6.86       |
|   SE9-16    | yolov9_opencv.py  | yolov9s_fp16_1b.bmodel  |      9.36       |      29.43      |      48.36      |      6.93       |
|   SE9-16    | yolov9_opencv.py  | yolov9s_int8_1b.bmodel  |      9.38       |      29.60      |      25.45      |      6.87       |
|   SE9-16    | yolov9_opencv.py  | yolov9s_int8_4b.bmodel  |      9.44       |      32.74      |      24.92      |      7.43       |
|   SE9-16    |  yolov9_bmcv.py   | yolov9s_fp32_1b.bmodel  |      4.29       |      4.59       |     165.88      |      6.96       |
|   SE9-16    |  yolov9_bmcv.py   | yolov9s_fp16_1b.bmodel  |      4.28       |      4.56       |      44.27      |      6.95       |
|   SE9-16    |  yolov9_bmcv.py   | yolov9s_int8_1b.bmodel  |      4.28       |      4.58       |      21.50      |      6.90       |
|   SE9-16    |  yolov9_bmcv.py   | yolov9s_int8_4b.bmodel  |      4.17       |      4.27       |      20.38      |      6.11       |
|   SE9-16    |  yolov9_bmcv.soc  | yolov9s_fp32_1b.bmodel  |      5.88       |      1.73       |     162.30      |      12.06      |
|   SE9-16    |  yolov9_bmcv.soc  | yolov9s_fp16_1b.bmodel  |      5.86       |      1.73       |      40.89      |      12.05      |
|   SE9-16    |  yolov9_bmcv.soc  | yolov9s_int8_1b.bmodel  |      5.90       |      1.74       |      18.17      |      12.06      |
|   SE9-16    |  yolov9_bmcv.soc  | yolov9s_int8_4b.bmodel  |      5.74       |      1.66       |      17.70      |      12.01      |
|   SE9-16    | yolov9_opencv.py  |yolov9s_fp32_1b_2core.bmodel|      9.50       |      29.87      |      98.46      |      6.91       |
|   SE9-16    | yolov9_opencv.py  |yolov9s_fp16_1b_2core.bmodel|      9.40       |      29.44      |      32.21      |      6.91       |
|   SE9-16    | yolov9_opencv.py  |yolov9s_int8_1b_2core.bmodel|      9.37       |      29.89      |      19.96      |      6.85       |
|   SE9-16    | yolov9_opencv.py  |yolov9s_int8_4b_2core.bmodel|      9.47       |      32.67      |      17.02      |      6.90       |
|   SE9-16    |  yolov9_bmcv.py   |yolov9s_fp32_1b_2core.bmodel|      4.32       |      4.59       |      94.68      |      6.96       |
|   SE9-16    |  yolov9_bmcv.py   |yolov9s_fp16_1b_2core.bmodel|      4.28       |      4.58       |      28.31      |      6.93       |
|   SE9-16    |  yolov9_bmcv.py   |yolov9s_int8_1b_2core.bmodel|      4.28       |      4.57       |      15.99      |      6.90       |
|   SE9-16    |  yolov9_bmcv.py   |yolov9s_int8_4b_2core.bmodel|      4.16       |      4.26       |      12.93      |      6.12       |
|   SE9-16    |  yolov9_bmcv.soc  |yolov9s_fp32_1b_2core.bmodel|      5.87       |      1.73       |      91.16      |      12.06      |
|   SE9-16    |  yolov9_bmcv.soc  |yolov9s_fp16_1b_2core.bmodel|      5.93       |      1.74       |      24.88      |      12.05      |
|   SE9-16    |  yolov9_bmcv.soc  |yolov9s_int8_1b_2core.bmodel|      5.85       |      1.73       |      12.63      |      12.06      |
|   SE9-16    |  yolov9_bmcv.soc  |yolov9s_int8_4b_2core.bmodel|      5.72       |      1.66       |      10.11      |      12.01      |
|    SE9-8    | yolov9_opencv.py  | yolov9s_fp32_1b.bmodel  |      9.42       |      30.22      |     169.91      |      7.02       |
|    SE9-8    | yolov9_opencv.py  | yolov9s_fp16_1b.bmodel  |      9.45       |      30.16      |      48.42      |      6.97       |
|    SE9-8    | yolov9_opencv.py  | yolov9s_int8_1b.bmodel  |      9.38       |      29.75      |      25.53      |      6.94       |
|    SE9-8    | yolov9_opencv.py  | yolov9s_int8_4b.bmodel  |      9.48       |      32.99      |      24.74      |      7.06       |
|    SE9-8    |  yolov9_bmcv.py   | yolov9s_fp32_1b.bmodel  |      4.19       |      4.48       |     165.90      |      7.04       |
|    SE9-8    |  yolov9_bmcv.py   | yolov9s_fp16_1b.bmodel  |      4.13       |      4.48       |      44.29      |      7.03       |
|    SE9-8    |  yolov9_bmcv.py   | yolov9s_int8_1b.bmodel  |      4.13       |      4.47       |      21.54      |      6.96       |
|    SE9-8    |  yolov9_bmcv.py   | yolov9s_int8_4b.bmodel  |      3.96       |      4.16       |      20.54      |      6.40       |
|    SE9-8    |  yolov9_bmcv.soc  | yolov9s_fp32_1b.bmodel  |      5.74       |      1.72       |     162.34      |      12.15      |
|    SE9-8    |  yolov9_bmcv.soc  | yolov9s_fp16_1b.bmodel  |      5.79       |      1.72       |      40.94      |      12.14      |
|    SE9-8    |  yolov9_bmcv.soc  | yolov9s_int8_1b.bmodel  |      5.75       |      1.72       |      18.15      |      12.15      |
|    SE9-8    |  yolov9_bmcv.soc  | yolov9s_int8_4b.bmodel  |      5.62       |      1.64       |      17.66      |      12.10      |
> **测试说明**：
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. SE5-16/SE7-32的主控处理器均为8核CA53@2.3GHz，SE9-16为8核CA53@1.6GHz，SE9-8为6核CA53@1.6GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异，不同的阈值对后处理时间影响较大。


## 8. FAQ
请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。
