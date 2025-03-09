# YOLOv8_plus_seg

## 目录

- [YOLOv8\_plus\_seg](#yolov8_plus_seg)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
    - [2.1 目录结构说明](#21-目录结构说明)
    - [2.2 SDK特性](#22-sdk特性)
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
YOLOv8_plus​_seg例程可以运行YOLOv8_seg系列模型，以及具有相同输入输出结构的衍生版本，目前已适配[​YOLOv8官方开源仓库](https://github.com/ultralytics/ultralytics)、[​YOLOv9官方开源仓库](https://github.com/WongKinYiu/yolov9)，支持在SOPHON BM1684X/BM1688上进行推理测试。

## 2. 特性

### 2.1 目录结构说明
```bash
├── cpp                   # 存放C++例程及其README
|   ├──README.md      
|   ├──yolov8_bmcv        # C++例程
├── docs                  # 存放本例程专用文档，如ONNX导出、移植常见问题等
├── pics                  # 存放README等说明文档中用到的图片
├── python                # 存放Python例程及其README
|   ├──README.md 
|   ├──yolov8_bmcv.py     # Python例程
|   └──...                # Python例程共用功能的封装。
├── README.md             # 本例程的中文指南
├── scripts               # 存放模型编译、数据下载、自动测试等shell脚本
└── tools                 # 存放精度测试、性能比对等python脚本
```

### 2.2 SDK特性
* 支持BM1688(SoC)和BM1684X(x86 PCIe、SoC、riscv PCIe)
* 支持FP32、FP16(BM1684X/BM1688)、INT8模型编译和推理
* 支持C++、Python推理
* 支持图片和视频测试

## 3. 数据准备与模型编译

### 3.1 数据准备

​本例程在`scripts`目录下提供了相关模型和数据的下载脚本`download.sh`，**如果您希望自己准备模型和数据集，可以跳过本小节，参考[3.2 模型编译](#32-模型编译)进行模型转换。**

```bash
chmod -R +x scripts/
./scripts/download.sh --all 
```

`download.sh`默认只下载`datasets`，`models`可以通过指定参数分平台下载，参数如下：
```bash
--all     # 下载所有模型
--BM1684X # 下载BM1684X的bmodel
--BM1688  # 下载BM1688的bmodel
--onnx    # 下载onnx
```

下载的模型包括：
```bash
models/
├── BM1684X # 在BM1684X上运行的模型
│   ├── yolov8s_fp16_1b.bmodel
│   ├── yolov8s_fp32_1b.bmodel
│   ├── yolov8s_int8_1b.bmodel
│   ├── yolov8s_int8_4b.bmodel
│   ├── yolov9c_fp16_1b.bmodel
│   ├── yolov9c_fp32_1b.bmodel
│   ├── yolov9c_int8_1b.bmodel
│   ├── yolov9c_int8_4b.bmodel
│   ├── yolov9s_fp16_1b.bmodel
│   └── yolov9s_fp32_1b.bmodel
├── BM1688 # 在BM1688上运行的模型
│   ├── yolov8s_fp16_1b.bmodel
│   ├── yolov8s_fp32_1b.bmodel
│   ├── yolov8s_int8_1b.bmodel
│   ├── yolov8s_int8_4b_2core.bmodel
│   ├── yolov8s_int8_4b.bmodel
│   ├── yolov9c_fp16_1b.bmodel
│   ├── yolov9c_fp32_1b.bmodel
│   ├── yolov9c_int8_1b.bmodel
│   ├── yolov9c_int8_4b_2core.bmodel
│   └── yolov9c_int8_4b.bmodel
├── onnx
    ├── yolov8s_qtable # 量化yolov8s-seg.onnx时，需要混合精度的层
    ├── yolov8s-seg.onnx
    ├── yolov9c_qtable # 量化yolov9s-c-seg-converted.onnx时，需要混合精度的层
    └── yolov9-c-seg-converted.onnx
```
下载的数据包括：
```bash
./datasets
├── test                                      # 测试图片
├── test_car_person_1080P.mp4                 # 测试视频
├── coco.names                                # coco类别名文件
├── coco128                                   # coco128数据集，用于模型量化
└── coco                                      
    ├── val2017_1000                               # coco val2017_1000数据集：coco val2017中随机抽取的1000张样本
    └── instances_val2017_1000.json                # coco val2017_1000数据集关键点标签文件，用于计算精度评价指标 
```

### 3.2 模型编译

**如果您不编译模型，只想直接使用下载的数据集和模型，可以跳过本小节。**

源模型需要编译成BModel才能在SOPHON TPU上运行，源模型在编译前要导出成onnx模型，如果您使用的TPU-MLIR版本>=v1.3.0（即官网v23.07.01），也可以直接使用torchscript模型。具体可参考[模型导出](./docs/YOLOv8_Export_Guide.md)。​同时，您需要准备用于测试的数据集，如果量化模型，还要准备用于量化的数据集。

建议使用TPU-MLIR编译BModel，模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录，并使用本例程提供的脚本将onnx模型编译为BModel。脚本中命令的详细说明可参考《TPU-MLIR开发手册》(请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的SDK中获取)。

- 生成FP32 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688**），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684x #bm1688
```

​执行上述命令会在`models/BM1684X`等文件夹下生成转换好的FP32 BModel。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688**），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x #bm1688
```

​执行上述命令会在`models/BM1684X/`等文件夹下生成转换好的FP16 BModel。

- 生成INT8 BModel

​本例程在`scripts`目录下提供了量化INT8 BModel的脚本，请注意修改`gen_int8bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台（**支持BM1684X/BM1688**），如：

```shell
./scripts/gen_int8bmodel_mlir.sh bm1684x #bm1688
```

​上述脚本会在`models/BM1684X`等文件夹下生成转换好的INT8 BModel。

注：这里用到了混合精度量化，需要将一些层设为敏感层，相应的qtable在此前`download.sh`下载的`models/onnx`文件夹里。如果您需要量化自己微调过的模型，可以参考[量化指南](../../docs/Calibration_Guide.md#13-特定模型优化技巧)中的方法，从我们提供的qtable倒推出自己模型需要的qtable。

## 4. 例程测试
- [C++例程](./cpp/README.md)
- [Python例程](./python/README.md)

## 5. 精度测试
### 5.1 测试方法

首先，参考[C++例程](cpp/README.md#32-测试图片)或[Python例程](python/README.md#22-测试图片)推理要测试的数据集，生成预测的json文件，注意修改数据集(datasets/coco/val2017_1000)和相关参数(conf_thresh=0.001、nms_thresh=0.7)。  
然后，使用`tools`目录下的`eval_coco.py`脚本，将测试生成的json文件与测试集标签json文件进行对比，计算出目标检测的评价指标，命令如下：
```bash
# 安装pycocotools，若已安装请跳过
pip3 install pycocotools
# 请根据实际情况修改程序路径和json文件路径
python3 tools/eval_coco.py --gt_path datasets/coco/instances_val2017_1000.json --result_json results/yolov8s_fp32_1b.bmodel_val2017_1000_opencv_python_result.json
```
### 5.2 测试结果
在coco2017 val数据集上，精度测试结果如下：
|   测试平台    |      测试程序     |      测试模型          |AP@IoU=0.5:0.95|AP@IoU=0.5|
| ------------ | ---------------- | ---------------------- | ------------- | -------- |
|   SE7-32    |  yolov8_bmcv.py   |      yolov8s_fp32_1b.bmodel       | 0.357 | 0.569 |
|   SE7-32    |  yolov8_bmcv.py   |      yolov8s_fp16_1b.bmodel       | 0.357 | 0.568 |
|   SE7-32    |  yolov8_bmcv.py   |      yolov8s_int8_1b.bmodel       | 0.350 | 0.554 |
|   SE7-32    |  yolov8_bmcv.py   |      yolov8s_int8_4b.bmodel       | 0.350 | 0.554 |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov8s_fp32_1b.bmodel       | 0.360 | 0.578 |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov8s_fp16_1b.bmodel       | 0.360 | 0.577 |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov8s_int8_1b.bmodel       | 0.354 | 0.567 |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov8s_int8_4b.bmodel       | 0.354 | 0.567 |
|   SE7-32    |  yolov8_bmcv.py   |      yolov9c_fp32_1b.bmodel       | 0.416 | 0.644 |
|   SE7-32    |  yolov8_bmcv.py   |      yolov9c_fp16_1b.bmodel       | 0.416 | 0.644 |
|   SE7-32    |  yolov8_bmcv.py   |      yolov9c_int8_1b.bmodel       | 0.382 | 0.598 |
|   SE7-32    |  yolov8_bmcv.py   |      yolov9c_int8_4b.bmodel       | 0.382 | 0.598 |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov9c_fp32_1b.bmodel       | 0.417 | 0.655 |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov9c_fp16_1b.bmodel       | 0.418 | 0.655 |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov9c_int8_1b.bmodel       | 0.388 | 0.615 |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov9c_int8_4b.bmodel       | 0.388 | 0.615 |
|   SE9-16    |  yolov8_bmcv.py   |      yolov8s_fp32_1b.bmodel       | 0.357 | 0.569 |
|   SE9-16    |  yolov8_bmcv.py   |      yolov8s_fp16_1b.bmodel       | 0.357 | 0.568 |
|   SE9-16    |  yolov8_bmcv.py   |      yolov8s_int8_1b.bmodel       | 0.350 | 0.554 |
|   SE9-16    |  yolov8_bmcv.py   |      yolov8s_int8_4b.bmodel       | 0.350 | 0.554 |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov8s_fp32_1b.bmodel       | 0.360 | 0.578 |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov8s_fp16_1b.bmodel       | 0.360 | 0.577 |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov8s_int8_1b.bmodel       | 0.355 | 0.567 |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov8s_int8_4b.bmodel       | 0.355 | 0.567 |
|   SE9-16    |  yolov8_bmcv.py   |   yolov8s_int8_4b_2core.bmodel    | 0.350 | 0.554 |
|   SE9-16    |  yolov8_bmcv.soc  |   yolov8s_int8_4b_2core.bmodel    | 0.355 | 0.567 |
|   SE9-16    |  yolov8_bmcv.py   |      yolov9c_fp32_1b.bmodel       | 0.416 | 0.644 |
|   SE9-16    |  yolov8_bmcv.py   |      yolov9c_fp16_1b.bmodel       | 0.416 | 0.644 |
|   SE9-16    |  yolov8_bmcv.py   |      yolov9c_int8_1b.bmodel       | 0.382 | 0.598 |
|   SE9-16    |  yolov8_bmcv.py   |      yolov9c_int8_4b.bmodel       | 0.382 | 0.598 |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov9c_fp32_1b.bmodel       | 0.418 | 0.655 |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov9c_fp16_1b.bmodel       | 0.418 | 0.655 |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov9c_int8_1b.bmodel       | 0.388 | 0.615 |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov9c_int8_4b.bmodel       | 0.388 | 0.615 |
|   SE9-16    |  yolov8_bmcv.py   |   yolov9c_int8_4b_2core.bmodel    | 0.382 | 0.598 |
|   SE9-16    |  yolov8_bmcv.soc  |   yolov9c_int8_4b_2core.bmodel    | 0.388 | 0.615 |

> **测试说明**：  
> 1. 由于sdk版本之间可能存在差异，实际运行结果与本表有<0.01的精度误差是正常的；
> 2. AP@IoU=0.5:0.95为area=all对应的指标；
> 3. 在搭载了相同TPU和SOPHONSDK的PCIe或SoC平台上，相同程序的精度一致，SE5系列对应BM1684，SE7系列对应BM1684X，SE9系列中，SE9-16对应BM1688；


## 6. 性能测试
### 6.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684X/yolov8s_fp32_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

|              测试模型           | calculate time(ms) |
| -------------------------------| ----------------- |
| BM1684X/yolov8s_fp32_1b.bmodel     |          41.59  |
| BM1684X/yolov8s_fp16_1b.bmodel     |           7.45  |
| BM1684X/yolov8s_int8_1b.bmodel     |           3.97  |
| BM1684X/yolov8s_int8_4b.bmodel     |           3.82  |
| BM1684X/yolov9c_fp32_1b.bmodel     |         137.30  |
| BM1684X/yolov9c_fp16_1b.bmodel     |          21.56  |
| BM1684X/yolov9c_int8_1b.bmodel     |          10.00  |
| BM1684X/yolov9c_int8_4b.bmodel     |           9.59  |
| BM1688/yolov8s_fp32_1b.bmodel      |         232.65  |
| BM1688/yolov8s_fp16_1b.bmodel      |          46.00  |
| BM1688/yolov8s_int8_1b.bmodel      |          10.69  |
| BM1688/yolov8s_int8_4b.bmodel      |          10.61  |
| BM1688/yolov8s_int8_4b_2core.bmodel|           6.49  |
| BM1688/yolov9c_fp32_1b.bmodel      |         781.42  |
| BM1688/yolov9c_fp16_1b.bmodel      |         148.13  |
| BM1688/yolov9c_int8_1b.bmodel      |          31.50  |
| BM1688/yolov9c_int8_4b.bmodel      |          31.06  |
| BM1688/yolov9c_int8_4b_2core.bmodel|          18.03  |

> **测试说明**：  
1. 性能测试结果具有一定的波动性；
2. `calculate time`已折算为平均每张图片的推理时间；
3. SoC和PCIe的测试结果基本一致。


### 6.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++例程打印的预处理时间、推理时间、后处理时间为整个batch处理的时间，需除以相应的batch size才是每张图片的处理时间。

在不同的测试平台上，使用不同的例程、模型测试`datasets/coco/val2017_1000`，conf_thresh=0.25，nms_thresh=0.7，性能测试结果如下：
|    测试平台  |     测试程序      |        测试模型        |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ---------------- | ---------------------- | -------- | --------- | --------- | --------- |
|   SE7-32    |  yolov8_bmcv.py   |      yolov8s_fp32_1b.bmodel       |      7.70       |      2.40       |      47.94      |      79.62      |
|   SE7-32    |  yolov8_bmcv.py   |      yolov8s_fp16_1b.bmodel       |      3.39       |      2.39       |      13.71      |      79.90      |
|   SE7-32    |  yolov8_bmcv.py   |      yolov8s_int8_1b.bmodel       |      3.38       |      2.38       |      10.34      |      77.68      |
|   SE7-32    |  yolov8_bmcv.py   |      yolov8s_int8_4b.bmodel       |      3.01       |      2.20       |      9.22       |      73.47      |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov8s_fp32_1b.bmodel       |      2.63       |      1.35       |      42.25      |      33.24      |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov8s_fp16_1b.bmodel       |      2.63       |      1.35       |      8.12       |      34.47      |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov8s_int8_1b.bmodel       |      2.62       |      1.35       |      4.61       |      31.42      |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov8s_int8_4b.bmodel       |      2.49       |      1.30       |      4.47       |      30.96      |
|   SE7-32    |  yolov8_bmcv.py   |      yolov9c_fp32_1b.bmodel       |      4.58       |      2.40       |     143.60      |      79.54      |
|   SE7-32    |  yolov8_bmcv.py   |      yolov9c_fp16_1b.bmodel       |      3.34       |      2.39       |      27.87      |      86.25      |
|   SE7-32    |  yolov8_bmcv.py   |      yolov9c_int8_1b.bmodel       |      3.35       |      2.39       |      16.29      |      97.64      |
|   SE7-32    |  yolov8_bmcv.py   |      yolov9c_int8_4b.bmodel       |      3.01       |      2.21       |      14.99      |      93.01      |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov9c_fp32_1b.bmodel       |      2.62       |      1.36       |     137.97      |      35.07      |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov9c_fp16_1b.bmodel       |      2.62       |      1.36       |      22.21      |      34.56      |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov9c_int8_1b.bmodel       |      2.63       |      1.35       |      10.65      |      38.48      |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov9c_int8_4b.bmodel       |      2.50       |      1.30       |      10.30      |      37.94      |
|   SE9-16    |  yolov8_bmcv.py   |      yolov8s_fp32_1b.bmodel       |      7.92       |      4.66       |     240.53      |      91.33      |
|   SE9-16    |  yolov8_bmcv.py   |      yolov8s_fp16_1b.bmodel       |      4.65       |      4.60       |      53.84      |      88.29      |
|   SE9-16    |  yolov8_bmcv.py   |      yolov8s_int8_1b.bmodel       |      4.61       |      4.55       |      18.69      |      84.55      |
|   SE9-16    |  yolov8_bmcv.py   |      yolov8s_int8_4b.bmodel       |      4.09       |      4.18       |      17.18      |      85.46      |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov8s_fp32_1b.bmodel       |      3.35       |      2.59       |     233.53      |      43.24      |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov8s_fp16_1b.bmodel       |      3.42       |      2.59       |      46.89      |      43.30      |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov8s_int8_1b.bmodel       |      3.40       |      2.59       |      11.60      |      40.85      |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov8s_int8_4b.bmodel       |      3.19       |      2.47       |      11.53      |      40.78      |
|   SE9-16    |  yolov8_bmcv.py   |   yolov8s_int8_4b_2core.bmodel    |      4.09       |      4.19       |      13.38      |      87.41      |
|   SE9-16    |  yolov8_bmcv.soc  |   yolov8s_int8_4b_2core.bmodel    |      3.16       |      2.48       |      7.41       |      40.71      |
|   SE9-16    |  yolov8_bmcv.py   |      yolov9c_fp32_1b.bmodel       |      4.72       |      4.66       |     789.65      |      92.48      |
|   SE9-16    |  yolov8_bmcv.py   |      yolov9c_fp16_1b.bmodel       |      4.67       |      4.67       |     156.08      |      92.12      |
|   SE9-16    |  yolov8_bmcv.py   |      yolov9c_int8_1b.bmodel       |      4.67       |      4.59       |      39.43      |     106.58      |
|   SE9-16    |  yolov8_bmcv.py   |      yolov9c_int8_4b.bmodel       |      4.11       |      4.16       |      37.73      |     103.82      |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov9c_fp32_1b.bmodel       |      3.38       |      2.59       |     782.38      |      45.98      |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov9c_fp16_1b.bmodel       |      3.39       |      2.60       |     149.05      |      45.31      |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov9c_int8_1b.bmodel       |      3.39       |      2.60       |      32.42      |      48.44      |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov9c_int8_4b.bmodel       |      3.18       |      2.47       |      31.99      |      48.66      |
|   SE9-16    |  yolov8_bmcv.py   |   yolov9c_int8_4b_2core.bmodel    |      4.11       |      4.18       |      24.77      |     104.64      |
|   SE9-16    |  yolov8_bmcv.soc  |   yolov9c_int8_4b_2core.bmodel    |      3.13       |      2.46       |      18.95      |      48.48      |

> **测试说明**：  
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. SE5-16/SE7-32的主控处理器均为8核CA53@2.3GHz，SE9-16为8核CA53@1.6GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异，不同的阈值对后处理时间影响较大。 


## 8. FAQ
请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。