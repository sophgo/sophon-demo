# YOLOv8_plus_det

## 目录

- [YOLOv8\_plus\_det](#yolov8_plus_det)
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
YOLOv8_plus​_det例程可以运行YOLOv8_det系列模型，以及具有相同输入输出结构的衍生版本，目前已适配[​YOLOv8官方开源仓库](https://github.com/ultralytics/ultralytics)、[​YOLOv9官方开源仓库](https://github.com/WongKinYiu/yolov9)、[<200b>YOLOv12官方开源仓库](https://github.com/sunsmarterjie/yolov12)，支持在SOPHON BM1684X/BM1688/CV186X上进行推理测试。

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
* 支持BM1688/CV186X(SoC)和BM1684X(x86 PCIe、SoC、riscv PCIe)
* 支持FP32、FP16(BM1684X/BM1688/CV186X)、INT8模型编译和推理
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
--CV186X  # 下载CV186X的bmodel
--onnx    # 下载onnx
```

下载的模型包括：
```bash
models/
├── BM1684X # 在BM1684X上运行的模型
│   ├── yolov12s_fp16_1b.bmodel
│   ├── yolov12s_fp32_1b.bmodel
│   ├── yolov12s_int8_1b.bmodel
│   ├── yolov12s_int8_4b.bmodel
│   ├── yolov11s_fp16_1b.bmodel
│   ├── yolov11s_fp32_1b.bmodel
│   ├── yolov11s_int8_1b.bmodel
│   ├── yolov11s_int8_4b.bmodel
│   ├── yolov8s_fp16_1b.bmodel
│   ├── yolov8s_fp32_1b.bmodel
│   ├── yolov8s_int8_1b.bmodel
│   ├── yolov8s_int8_4b.bmodel
│   ├── yolov9s_fp16_1b.bmodel
│   ├── yolov9s_fp32_1b.bmodel
│   ├── yolov9s_int8_1b.bmodel
│   └── yolov9s_int8_4b.bmodel
├── BM1688 # 在BM1688上运行的模型
│   ├── yolov12s_fp16_1b.bmodel
│   ├── yolov12s_fp32_1b.bmodel
│   ├── yolov12s_int8_1b.bmodel
│   ├── yolov12s_int8_4b_2core.bmodel
│   ├── yolov12s_int8_4b.bmodel
│   ├── yolov11s_fp16_1b.bmodel
│   ├── yolov11s_fp32_1b.bmodel
│   ├── yolov11s_int8_1b.bmodel
│   ├── yolov11s_int8_4b_2core.bmodel
│   ├── yolov11s_int8_4b.bmodel
│   ├── yolov8s_fp16_1b.bmodel
│   ├── yolov8s_fp32_1b.bmodel
│   ├── yolov8s_int8_1b.bmodel
│   ├── yolov8s_int8_4b_2core.bmodel
│   ├── yolov8s_int8_4b.bmodel
│   ├── yolov9s_fp16_1b.bmodel
│   ├── yolov9s_fp32_1b.bmodel
│   ├── yolov9s_int8_1b.bmodel
│   ├── yolov9s_int8_4b_2core.bmodel
│   └── yolov9s_int8_4b.bmodel
├── CV186X
│   ├── yolov12s_fp16_1b.bmodel
│   ├── yolov12s_fp32_1b.bmodel
│   ├── yolov12s_int8_1b.bmodel
│   ├── yolov12s_int8_4b.bmodel
│   ├── yolov11s_fp16_1b.bmodel
│   ├── yolov11s_fp32_1b.bmodel
│   ├── yolov11s_int8_1b.bmodel
│   ├── yolov11s_int8_4b.bmodel
│   ├── yolov8s_fp16_1b.bmodel
│   ├── yolov8s_fp32_1b.bmodel
│   ├── yolov8s_int8_1b.bmodel
│   ├── yolov8s_int8_4b.bmodel
│   ├── yolov9s_fp16_1b.bmodel
│   ├── yolov9s_fp32_1b.bmodel
│   ├── yolov9s_int8_1b.bmodel
│   └── yolov9s_int8_4b.bmodel
├── onnx
    ├── yolov12s.onnx
    ├── yolov12s_qtable # 量化yolov12s.onnx时，需要混合精度的层
    ├── yolo11s.onnx
    ├── yolov11s_qtable # 量化yolov11s.onnx时，需要混合精度的层
    ├── yolov8s.onnx
    ├── yolov8s_qtable # 量化yolov8s.onnx时，需要混合精度的层
    ├── yolov9s_qtable # 量化yolov9-s-converted.onnx时，需要混合精度的层
    └── yolov9-s-converted.onnx
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

​本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684x #bm1688/cv186x
```

​执行上述命令会在`models/BM1684X`等文件夹下生成转换好的FP32 BModel。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688**），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x #bm1688/cv186x
```

​执行上述命令会在`models/BM1684X/`等文件夹下生成转换好的FP16 BModel。

- 生成INT8 BModel

​本例程在`scripts`目录下提供了量化INT8 BModel的脚本，请注意修改`gen_int8bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台（**支持BM1684X/BM1688**），如：

```shell
./scripts/gen_int8bmodel_mlir.sh bm1684x #bm1688/cv186x
```

​上述脚本会在`models/BM1684X`等文件夹下生成转换好的INT8 BModel。

注：这里用到了混合精度量化，需要将一些层设为敏感层，相应的qtable在此前`download.sh`下载的`models/onnx`文件夹里。如果您需要量化自己微调过的模型，可以参考[量化指南](../../docs/Calibration_Guide.md#13-特定模型优化技巧)中的方法，从我们提供的qtable倒推出自己模型需要的qtable。BM1684不支持F16混合精度，如果您使用BM1684系列产品，您需要把qtable中的F16层更改为F32。

## 4. 例程测试
- [C++例程](./cpp/README.md)
- [Python例程](./python/README.md)

## 5. 精度测试
### 5.1 测试方法

首先，参考[C++例程](cpp/README.md#32-测试图片)或[Python例程](python/README.md#22-测试图片)推理要测试的数据集，生成预测的json文件，注意修改数据集(datasets/coco/val2017_1000)和相关参数(conf_thresh=0.001、nms_thresh=0.7)。  
然后，使用`tools`目录下的`eval_coco.py`脚本，将测试生成的json文件与测试集标签json文件进行对比，计算出目标检测的评价指标，命令如下：
```bash
# 安装pycocotools，若已安装请跳过
pip3 install pycocotools==2.0.8
# 请根据实际情况修改程序路径和json文件路径
python3 tools/eval_coco.py --gt_path datasets/coco/instances_val2017_1000.json --result_json results/yolov8s_fp32_1b.bmodel_val2017_1000_bmcv_python_result.json
```
### 5.2 测试结果
在coco2017 val数据集上，精度测试结果如下：
|   测试平台    |      测试程序     |      测试模型          |AP@IoU=0.5:0.95|AP@IoU=0.5|
| ------------ | ---------------- | ---------------------- | ------------- | -------- |
|   SE7-32    |  yolov8_bmcv.py   |      yolov8s_fp32_1b.bmodel       | 0.447 | 0.610 |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov8s_fp32_1b.bmodel       | 0.453 | 0.620 |
|   SE7-32    |  yolov8_bmcv.py   |      yolov8s_fp16_1b.bmodel       | 0.447 | 0.610 |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov8s_fp16_1b.bmodel       | 0.453 | 0.620 |
|   SE7-32    |  yolov8_bmcv.py   |      yolov8s_int8_1b.bmodel       | 0.442 | 0.607 |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov8s_int8_1b.bmodel       | 0.449 | 0.617 |
|   SE7-32    |  yolov8_bmcv.py   |      yolov8s_int8_4b.bmodel       | 0.442 | 0.607 |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov8s_int8_4b.bmodel       | 0.449 | 0.617 |
|   SE7-32    |  yolov8_bmcv.py   |      yolov9s_fp32_1b.bmodel       | 0.464 | 0.630 |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov9s_fp32_1b.bmodel       | 0.468 | 0.636 |
|   SE7-32    |  yolov8_bmcv.py   |      yolov9s_fp16_1b.bmodel       | 0.463 | 0.630 |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov9s_fp16_1b.bmodel       | 0.469 | 0.637 |
|   SE7-32    |  yolov8_bmcv.py   |      yolov9s_int8_1b.bmodel       | 0.455 | 0.624 |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov9s_int8_1b.bmodel       | 0.460 | 0.632 |
|   SE7-32    |  yolov8_bmcv.py   |      yolov9s_int8_4b.bmodel       | 0.455 | 0.624 |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov9s_int8_4b.bmodel       | 0.460 | 0.632 |
|   SE7-32    |  yolov8_bmcv.py   |      yolov11s_fp32_1b.bmodel      | 0.471 | 0.638 |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov11s_fp32_1b.bmodel      | 0.474 | 0.645 |
|   SE7-32    |  yolov8_bmcv.py   |      yolov11s_fp16_1b.bmodel      | 0.470 | 0.638 |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov11s_fp16_1b.bmodel      | 0.475 | 0.645 |
|   SE7-32    |  yolov8_bmcv.py   |      yolov11s_int8_1b.bmodel      | 0.462 | 0.628 |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov11s_int8_1b.bmodel      | 0.468 | 0.638 |
|   SE7-32    |  yolov8_bmcv.py   |      yolov11s_int8_4b.bmodel      | 0.462 | 0.628 |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov11s_int8_4b.bmodel      | 0.468 | 0.638 |
|   SE7-32    |  yolov8_bmcv.py   |      yolov12s_fp32_1b.bmodel      | 0.474 | 0.640 |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov12s_fp32_1b.bmodel      | 0.481 | 0.650 |
|   SE7-32    |  yolov8_bmcv.py   |      yolov12s_fp16_1b.bmodel      | 0.474 | 0.640 |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov12s_fp16_1b.bmodel      | 0.481 | 0.651 |
|   SE7-32    |  yolov8_bmcv.py   |      yolov12s_int8_1b.bmodel      | 0.468 | 0.633 |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov12s_int8_1b.bmodel      | 0.473 | 0.642 |
|   SE7-32    |  yolov8_bmcv.py   |      yolov12s_int8_4b.bmodel      | 0.468 | 0.633 |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov12s_int8_4b.bmodel      | 0.473 | 0.642 |
|   SE9-16    |  yolov8_bmcv.py   |      yolov8s_fp32_1b.bmodel       | 0.447 | 0.610 |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov8s_fp32_1b.bmodel       | 0.453 | 0.620 |
|   SE9-16    |  yolov8_bmcv.py   |      yolov8s_fp16_1b.bmodel       | 0.447 | 0.610 |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov8s_fp16_1b.bmodel       | 0.453 | 0.620 |
|   SE9-16    |  yolov8_bmcv.py   |      yolov8s_int8_1b.bmodel       | 0.442 | 0.607 |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov8s_int8_1b.bmodel       | 0.450 | 0.618 |
|   SE9-16    |  yolov8_bmcv.py   |      yolov8s_int8_4b.bmodel       | 0.442 | 0.607 |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov8s_int8_4b.bmodel       | 0.450 | 0.618 |
|   SE9-16    |  yolov8_bmcv.py   |      yolov8s_int8_4b_2core.bmodel | 0.442 | 0.607 |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov8s_int8_4b_2core.bmodel | 0.450 | 0.618 |
|   SE9-16    |  yolov8_bmcv.py   |      yolov9s_fp32_1b.bmodel       | 0.465 | 0.630 |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov9s_fp32_1b.bmodel       | 0.469 | 0.637 |
|   SE9-16    |  yolov8_bmcv.py   |      yolov9s_fp16_1b.bmodel       | 0.463 | 0.630 |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov9s_fp16_1b.bmodel       | 0.469 | 0.637 |
|   SE9-16    |  yolov8_bmcv.py   |      yolov9s_int8_1b.bmodel       | 0.454 | 0.623 |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov9s_int8_1b.bmodel       | 0.459 | 0.631 |
|   SE9-16    |  yolov8_bmcv.py   |      yolov9s_int8_4b.bmodel       | 0.454 | 0.623 |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov9s_int8_4b.bmodel       | 0.459 | 0.631 |
|   SE9-16    |  yolov8_bmcv.py   |      yolov9s_int8_4b_2core.bmodel | 0.454 | 0.623 |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov9s_int8_4b_2core.bmodel | 0.459 | 0.631 |
|   SE9-16    |  yolov8_bmcv.py   |      yolov11s_fp32_1b.bmodel      | 0.471 | 0.638 |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov11s_fp32_1b.bmodel      | 0.476 | 0.645 |
|   SE9-16    |  yolov8_bmcv.py   |      yolov11s_fp16_1b.bmodel      | 0.471 | 0.638 |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov11s_fp16_1b.bmodel      | 0.475 | 0.645 |
|   SE9-16    |  yolov8_bmcv.py   |      yolov11s_int8_1b.bmodel      | 0.463 | 0.629 |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov11s_int8_1b.bmodel      | 0.468 | 0.638 |
|   SE9-16    |  yolov8_bmcv.py   |      yolov11s_int8_4b.bmodel      | 0.463 | 0.629 |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov11s_int8_4b.bmodel      | 0.468 | 0.638 |
|   SE9-16    |  yolov8_bmcv.py   |      yolov11s_int8_4b_2core.bmodel| 0.463 | 0.629 |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov11s_int8_4b_2core.bmodel| 0.468 | 0.638 |
|   SE9-16    |  yolov8_bmcv.py   |      yolov12s_fp32_1b.bmodel      | 0.474 | 0.640 |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov12s_fp32_1b.bmodel      | 0.480 | 0.650 |
|   SE9-16    |  yolov8_bmcv.py   |      yolov12s_fp16_1b.bmodel      | 0.474 | 0.640 |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov12s_fp16_1b.bmodel      | 0.480 | 0.651 |
|   SE9-16    |  yolov8_bmcv.py   |      yolov12s_int8_1b.bmodel      | 0.466 | 0.631 |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov12s_int8_1b.bmodel      | 0.472 | 0.641 |
|   SE9-16    |  yolov8_bmcv.py   |      yolov12s_int8_4b.bmodel      | 0.466 | 0.631 |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov12s_int8_4b.bmodel      | 0.472 | 0.641 |
|   SE9-16    |  yolov8_bmcv.py   |      yolov12s_int8_4b_core.bmodel | 0.466 | 0.631 |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov12s_int8_4b_core.bmodel | 0.472 | 0.641 |
|    SE9-8    |  yolov8_bmcv.py   |      yolov8s_fp32_1b.bmodel       | 0.447 | 0.610 |
|    SE9-8    |  yolov8_bmcv.soc  |      yolov8s_fp32_1b.bmodel       | 0.453 | 0.620 |
|    SE9-8    |  yolov8_bmcv.py   |      yolov8s_fp16_1b.bmodel       | 0.447 | 0.610 |
|    SE9-8    |  yolov8_bmcv.soc  |      yolov8s_fp16_1b.bmodel       | 0.453 | 0.620 |
|    SE9-8    |  yolov8_bmcv.py   |      yolov8s_int8_1b.bmodel       | 0.442 | 0.607 |
|    SE9-8    |  yolov8_bmcv.soc  |      yolov8s_int8_1b.bmodel       | 0.450 | 0.618 |
|    SE9-8    |  yolov8_bmcv.py   |      yolov8s_int8_4b.bmodel       | 0.442 | 0.607 |
|    SE9-8    |  yolov8_bmcv.soc  |      yolov8s_int8_4b.bmodel       | 0.450 | 0.618 |
|    SE9-8    |  yolov8_bmcv.py   |      yolov9s_fp32_1b.bmodel       | 0.465 | 0.630 |
|    SE9-8    |  yolov8_bmcv.soc  |      yolov9s_fp32_1b.bmodel       | 0.469 | 0.637 |
|    SE9-8    |  yolov8_bmcv.py   |      yolov9s_fp16_1b.bmodel       | 0.463 | 0.630 |
|    SE9-8    |  yolov8_bmcv.soc  |      yolov9s_fp16_1b.bmodel       | 0.469 | 0.637 |
|    SE9-8    |  yolov8_bmcv.py   |      yolov9s_int8_1b.bmodel       | 0.454 | 0.623 |
|    SE9-8    |  yolov8_bmcv.soc  |      yolov9s_int8_1b.bmodel       | 0.459 | 0.631 |
|    SE9-8    |  yolov8_bmcv.py   |      yolov9s_int8_4b.bmodel       | 0.454 | 0.623 |
|    SE9-8    |  yolov8_bmcv.soc  |      yolov9s_int8_4b.bmodel       | 0.459 | 0.631 |
|    SE9-8    |  yolov8_bmcv.py   |      yolov11s_fp32_1b.bmodel      | 0.471 | 0.638 |
|    SE9-8    |  yolov8_bmcv.soc  |      yolov11s_fp32_1b.bmodel      | 0.476 | 0.645 |
|    SE9-8    |  yolov8_bmcv.py   |      yolov11s_fp16_1b.bmodel      | 0.471 | 0.638 |
|    SE9-8    |  yolov8_bmcv.soc  |      yolov11s_fp16_1b.bmodel      | 0.475 | 0.645 |
|    SE9-8    |  yolov8_bmcv.py   |      yolov11s_int8_1b.bmodel      | 0.463 | 0.629 |
|    SE9-8    |  yolov8_bmcv.soc  |      yolov11s_int8_1b.bmodel      | 0.468 | 0.638 |
|    SE9-8    |  yolov8_bmcv.py   |      yolov11s_int8_4b.bmodel      | 0.463 | 0.629 |
|    SE9-8    |  yolov8_bmcv.soc  |      yolov11s_int8_4b.bmodel      | 0.468 | 0.638 |
|    SE9-8    |  yolov8_bmcv.py   |      yolov12s_fp32_1b.bmodel      | 0.474 | 0.640 |
|    SE9-8    |  yolov8_bmcv.soc  |      yolov12s_fp32_1b.bmodel      | 0.480 | 0.650 |
|    SE9-8    |  yolov8_bmcv.py   |      yolov12s_fp16_1b.bmodel      | 0.474 | 0.640 |
|    SE9-8    |  yolov8_bmcv.soc  |      yolov12s_fp16_1b.bmodel      | 0.480 | 0.651 |
|    SE9-8    |  yolov8_bmcv.py   |      yolov12s_int8_1b.bmodel      | 0.466 | 0.631 |
|    SE9-8    |  yolov8_bmcv.soc  |      yolov12s_int8_1b.bmodel      | 0.472 | 0.641 |
|    SE9-8    |  yolov8_bmcv.py   |      yolov12s_int8_4b.bmodel      | 0.466 | 0.631 |
|    SE9-8    |  yolov8_bmcv.soc  |      yolov12s_int8_4b.bmodel      | 0.472 | 0.641 |

> **测试说明**：  
> 1. 由于sdk版本之间可能存在差异，实际运行结果与本表有<0.01的精度误差是正常的；
> 2. AP@IoU=0.5:0.95为area=all对应的指标；
> 3. 在搭载了相同TPU和SOPHONSDK的PCIe或SoC平台上，相同程序的精度一致，SE5系列对应BM1684，SE7系列对应BM1684X，SE9系列中，SE9-16对应BM1688，SE9-8对应CV186X；

## 6. 性能测试
### 6.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684X/yolov8s_fp32_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

|    测试平台  |              测试模型           | calculate time(ms) |
| ----------- | -------------------------------| ----------------- |
|   SE7-32    | BM1684X/yolov8s_fp32_1b.bmodel     |          29.29  |
|   SE7-32    | BM1684X/yolov8s_fp16_1b.bmodel     |           5.59  |
|   SE7-32    | BM1684X/yolov8s_int8_1b.bmodel     |           2.92  |
|   SE7-32    | BM1684X/yolov8s_int8_4b.bmodel     |           2.80  |
|   SE7-32    | BM1684X/yolov9s_fp32_1b.bmodel     |          33.51  |
|   SE7-32    | BM1684X/yolov9s_fp16_1b.bmodel     |           7.41  |
|   SE7-32    | BM1684X/yolov9s_int8_1b.bmodel     |           4.83  |
|   SE7-32    | BM1684X/yolov9s_int8_4b.bmodel     |           4.59  |
|   SE7-32    | BM1684X/yolov11s_fp32_1b.bmodel    |          24.62  |
|   SE7-32    | BM1684X/yolov11s_fp16_1b.bmodel    |           5.88  |
|   SE7-32    | BM1684X/yolov11s_int8_1b.bmodel    |           3.27  |
|   SE7-32    | BM1684X/yolov11s_int8_4b.bmodel    |           3.01  |
|   SE7-32    | BM1684X/yolov12s_fp32_1b.bmodel    |          54.50  |
|   SE7-32    | BM1684X/yolov12s_fp16_1b.bmodel    |          26.90  |
|   SE7-32    | BM1684X/yolov12s_int8_1b.bmodel    |          22.33  |
|   SE7-32    | BM1684X/yolov12s_int8_4b.bmodel    |          21.56  |
|   SE9-16    | BM1688/yolov8s_fp32_1b.bmodel      |         161.70  |
|   SE9-16    | BM1688/yolov8s_fp16_1b.bmodel      |          34.63  |
|   SE9-16    | BM1688/yolov8s_int8_1b.bmodel      |           7.68  |
|   SE9-16    | BM1688/yolov8s_int8_4b.bmodel      |           7.54  |
|   SE9-16    | BM1688/yolov8s_int8_4b_2core.bmodel|           4.81  |
|   SE9-16    | BM1688/yolov9s_fp32_1b.bmodel      |         162.37  |
|   SE9-16    | BM1688/yolov9s_fp16_1b.bmodel      |          41.21  |
|   SE9-16    | BM1688/yolov9s_int8_1b.bmodel      |          18.03  |
|   SE9-16    | BM1688/yolov9s_int8_4b.bmodel      |          17.78  |
|   SE9-16    | BM1688/yolov9s_int8_4b_2core.bmodel|          10.11  |
|   SE9-16    | BM1688/yolov11s_fp32_1b.bmodel     |         131.64  |
|   SE9-16    | BM1688/yolov11s_fp16_1b.bmodel     |          33.87  |
|   SE9-16    | BM1688/yolov11s_int8_1b.bmodel     |           8.37  |
|   SE9-16    | BM1688/yolov11s_int8_4b.bmodel     |           7.98  |
|   SE9-16    | BM1688/yolov11s_int8_4b_2core.bmodel|           5.32  |
|   SE9-16    | BM1688/yolov12s_fp32_1b.bmodel     |         209.90  |
|   SE9-16    | BM1688/yolov12s_fp16_1b.bmodel     |          68.92  |
|   SE9-16    | BM1688/yolov12s_int8_1b.bmodel     |          59.99  |
|   SE9-16    | BM1688/yolov12s_int8_4b.bmodel     |          60.16  |
|   SE9-16    | BM1688/yolov12s_int8_4b_2core.bmodel|          37.70  |
|   SE9-8    | CV186X/yolov8s_fp32_1b.bmodel      |         165.35  |
|   SE9-8    | CV186X/yolov8s_fp16_1b.bmodel      |          36.80  |
|   SE9-8    | CV186X/yolov8s_int8_1b.bmodel      |           9.21  |
|   SE9-8    | CV186X/yolov8s_int8_4b.bmodel      |           9.14  |
|   SE9-8    | CV186X/yolov11s_fp32_1b.bmodel     |         135.15  |
|   SE9-8    | CV186X/yolov11s_fp16_1b.bmodel     |          36.22  |
|   SE9-8    | CV186X/yolov11s_int8_1b.bmodel     |          10.60  |
|   SE9-8    | CV186X/yolov11s_int8_4b.bmodel     |          10.10  |
|   SE9-8    | CV186X/yolov12s_fp32_1b.bmodel     |         209.44  |
|   SE9-8    | CV186X/yolov12s_fp16_1b.bmodel     |          68.68  |
|   SE9-8    | CV186X/yolov12s_int8_1b.bmodel     |          59.78  |
|   SE9-8    | CV186X/yolov12s_int8_4b.bmodel     |          59.96  |

> **测试说明**：  
1. 性能测试结果具有一定的波动性；
2. `calculate time`已折算为平均每张图片的推理时间；
3. SoC和PCIe的测试结果基本一致。


### 6.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++和Python例程打印的时间已经折算为单张图片的处理时间。

在不同的测试平台上，使用不同的例程、模型测试`datasets/coco/val2017_1000`，conf_thresh=0.25，nms_thresh=0.7，性能测试结果如下：
|    测试平台  |     测试程序      |        测试模型        |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ---------------- | ---------------------- | -------- | --------- | --------- | --------- |
|   SE7-32    |  yolov8_bmcv.py   |      yolov8s_fp32_1b.bmodel       |      3.16       |      2.39       |      31.99      |      5.88       |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov8s_fp32_1b.bmodel       |      2.58       |      1.37       |      29.53      |      3.23       |
|   SE7-32    |  yolov8_bmcv.py   |      yolov8s_fp16_1b.bmodel       |      3.17       |      2.40       |      8.29       |      5.93       |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov8s_fp16_1b.bmodel       |      2.59       |      1.37       |      5.82       |      3.23       |
|   SE7-32    |  yolov8_bmcv.py   |      yolov8s_int8_1b.bmodel       |      3.20       |      2.41       |      5.63       |      5.86       |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov8s_int8_1b.bmodel       |      2.59       |      1.37       |      3.13       |      3.24       |
|   SE7-32    |  yolov8_bmcv.py   |      yolov8s_int8_4b.bmodel       |      2.98       |      2.20       |      4.97       |      5.31       |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov8s_int8_4b.bmodel       |      2.47       |      1.31       |      3.06       |      3.22       |
|   SE7-32    |  yolov8_bmcv.py   |      yolov9s_fp32_1b.bmodel       |      3.17       |      2.38       |      36.20      |      5.88       |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov9s_fp32_1b.bmodel       |      2.59       |      1.37       |      33.73      |      3.23       |
|   SE7-32    |  yolov8_bmcv.py   |      yolov9s_fp16_1b.bmodel       |      3.15       |      2.40       |      10.09      |      5.87       |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov9s_fp16_1b.bmodel       |      2.59       |      1.37       |      7.65       |      3.23       |
|   SE7-32    |  yolov8_bmcv.py   |      yolov9s_int8_1b.bmodel       |      3.16       |      2.39       |      7.47       |      5.90       |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov9s_int8_1b.bmodel       |      2.60       |      1.37       |      5.04       |      3.23       |
|   SE7-32    |  yolov8_bmcv.py   |      yolov9s_int8_4b.bmodel       |      2.99       |      2.20       |      6.78       |      5.31       |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov9s_int8_4b.bmodel       |      2.48       |      1.32       |      4.86       |      3.21       |
|   SE7-32    |  yolov8_bmcv.py   |      yolov11s_fp32_1b.bmodel      |      3.15       |      2.40       |      27.34      |      5.87       |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov11s_fp32_1b.bmodel      |      2.59       |      1.37       |      24.86      |      3.23       |
|   SE7-32    |  yolov8_bmcv.py   |      yolov11s_fp16_1b.bmodel      |      3.16       |      2.39       |      8.58       |      5.88       |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov11s_fp16_1b.bmodel      |      2.60       |      1.37       |      6.12       |      3.24       |
|   SE7-32    |  yolov8_bmcv.py   |      yolov11s_int8_1b.bmodel      |      3.16       |      2.39       |      5.94       |      5.79       |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov11s_int8_1b.bmodel      |      2.60       |      1.37       |      3.50       |      3.23       |
|   SE7-32    |  yolov8_bmcv.py   |      yolov11s_int8_4b.bmodel      |      2.98       |      2.21       |      5.21       |      5.23       |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov11s_int8_4b.bmodel      |      2.46       |      1.32       |      3.27       |      3.21       |
|   SE7-32    |  yolov8_bmcv.py   |      yolov12s_fp32_1b.bmodel      |      3.17       |      2.40       |      57.73      |      5.81       |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov12s_fp32_1b.bmodel      |      2.59       |      1.37       |      55.27      |      3.24       |
|   SE7-32    |  yolov8_bmcv.py   |      yolov12s_fp16_1b.bmodel      |      3.16       |      2.40       |      29.84      |      5.79       |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov12s_fp16_1b.bmodel      |      2.59       |      1.37       |      27.38      |      3.24       |
|   SE7-32    |  yolov8_bmcv.py   |      yolov12s_int8_1b.bmodel      |      3.16       |      2.39       |      25.26      |      5.73       |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov12s_int8_1b.bmodel      |      2.59       |      1.37       |      22.83      |      3.23       |
|   SE7-32    |  yolov8_bmcv.py   |      yolov12s_int8_4b.bmodel      |      2.99       |      2.21       |      24.17      |      5.21       |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov12s_int8_4b.bmodel      |      2.47       |      1.32       |      22.19      |      3.22       |
|   SE9-16    |  yolov8_bmcv.py   |      yolov8s_fp32_1b.bmodel       |      4.18       |      4.53       |     165.23      |      7.29       |
|   SE9-16    |  yolov8_bmcv.py   |      yolov8s_fp16_1b.bmodel       |      4.20       |      4.47       |      38.08      |      7.30       |
|   SE9-16    |  yolov8_bmcv.py   |      yolov8s_int8_1b.bmodel       |      4.21       |      4.49       |      11.08      |      7.46       |
|   SE9-16    |  yolov8_bmcv.py   |      yolov8s_int8_4b.bmodel       |      3.99       |      4.17       |      10.24      |      6.51       |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov8s_fp32_1b.bmodel       |      3.28       |      2.60       |     162.00      |      4.06       |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov8s_fp16_1b.bmodel       |      3.31       |      2.58       |      34.93      |      4.03       |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov8s_int8_1b.bmodel       |      3.28       |      2.58       |      7.99       |      4.03       |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov8s_int8_4b.bmodel       |      3.09       |      2.46       |      7.90       |      3.98       |
|   SE9-16    |  yolov8_bmcv.py   |   yolov8s_int8_4b_2core.bmodel    |      3.98       |      4.19       |      7.75       |      6.50       |
|   SE9-16    |  yolov8_bmcv.soc  |   yolov8s_int8_4b_2core.bmodel    |      3.10       |      2.46       |      5.17       |      3.99       |
|   SE9-16    |  yolov8_bmcv.py   |      yolov9s_fp32_1b.bmodel       |      4.19       |      4.51       |     165.90      |      7.28       |
|   SE9-16    |  yolov8_bmcv.py   |      yolov9s_fp16_1b.bmodel       |      4.19       |      4.47       |      44.67      |      7.30       |
|   SE9-16    |  yolov8_bmcv.py   |      yolov9s_int8_1b.bmodel       |      4.18       |      4.47       |      21.49      |      7.29       |
|   SE9-16    |  yolov8_bmcv.py   |      yolov9s_int8_4b.bmodel       |      3.98       |      4.18       |      20.57      |      6.51       |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov9s_fp32_1b.bmodel       |      3.25       |      2.59       |     162.66      |      4.05       |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov9s_fp16_1b.bmodel       |      3.29       |      2.58       |      41.53      |      4.04       |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov9s_int8_1b.bmodel       |      3.29       |      2.58       |      18.34      |      4.04       |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov9s_int8_4b.bmodel       |      3.11       |      2.46       |      18.14      |      3.99       |
|   SE9-16    |  yolov8_bmcv.py   |   yolov9s_int8_4b_2core.bmodel    |      3.96       |      4.17       |      12.82      |      6.51       |
|   SE9-16    |  yolov8_bmcv.soc  |   yolov9s_int8_4b_2core.bmodel    |      3.09       |      2.46       |      10.47      |      3.99       |
|   SE9-16    |  yolov8_bmcv.py   |      yolov11s_fp32_1b.bmodel      |      4.21       |      4.49       |     135.11      |      7.25       |
|   SE9-16    |  yolov8_bmcv.py   |      yolov11s_fp16_1b.bmodel      |      4.17       |      4.50       |      37.24      |      7.27       |
|   SE9-16    |  yolov8_bmcv.py   |      yolov11s_int8_1b.bmodel      |      4.20       |      4.48       |      11.83      |      7.18       |
|   SE9-16    |  yolov8_bmcv.py   |      yolov11s_int8_4b.bmodel      |      4.00       |      4.19       |      10.75      |      6.40       |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov11s_fp32_1b.bmodel      |      3.27       |      2.60       |     131.90      |      4.06       |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov11s_fp16_1b.bmodel      |      3.29       |      2.58       |      34.18      |      4.04       |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov11s_int8_1b.bmodel      |      3.29       |      2.59       |      8.68       |      4.02       |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov11s_int8_4b.bmodel      |      3.10       |      2.46       |      8.33       |      3.98       |
|   SE9-16    |  yolov8_bmcv.py   |   yolov11s_int8_4b_2core.bmodel   |      3.99       |      4.18       |      8.23       |      6.40       |
|   SE9-16    |  yolov8_bmcv.soc  |   yolov11s_int8_4b_2core.bmodel   |      3.11       |      2.46       |      5.68       |      3.99       |
|   SE9-16    |  yolov8_bmcv.py   |      yolov12s_fp32_1b.bmodel      |      8.54       |      4.52       |     213.61      |      7.60       |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov12s_fp32_1b.bmodel      |      3.21       |      2.61       |     210.32      |      4.08       |
|   SE9-16    |  yolov8_bmcv.py   |      yolov12s_fp16_1b.bmodel      |      5.30       |      4.51       |      72.61      |      7.58       |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov12s_fp16_1b.bmodel      |      3.25       |      2.61       |      69.31      |      4.05       |
|   SE9-16    |  yolov8_bmcv.py   |      yolov12s_int8_1b.bmodel      |      5.83       |      4.50       |      63.46      |      7.48       |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov12s_int8_1b.bmodel      |      3.27       |      2.61       |      60.38      |      4.05       |
|   SE9-16    |  yolov8_bmcv.py   |      yolov12s_int8_4b.bmodel      |      5.35       |      4.16       |      62.94      |      6.62       |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov12s_int8_4b.bmodel      |      3.06       |      2.49       |      60.57      |      4.00       |
|   SE9-16    |  yolov8_bmcv.py   |   yolov12s_int8_4b_2core.bmodel   |      5.55       |      4.16       |      40.36      |      6.63       |
|   SE9-16    |  yolov8_bmcv.soc  |   yolov12s_int8_4b_2core.bmodel   |      3.08       |      2.49       |      37.96      |      4.01       |
|    SE9-8    |  yolov8_bmcv.py   |      yolov8s_fp32_1b.bmodel       |      4.22       |      4.51       |     168.83      |      7.57       |
|    SE9-8    |  yolov8_bmcv.py   |      yolov8s_fp16_1b.bmodel       |      4.25       |      4.50       |      40.15      |      7.66       |
|    SE9-8    |  yolov8_bmcv.py   |      yolov8s_int8_1b.bmodel       |      4.21       |      4.48       |      12.62      |      7.59       |
|    SE9-8    |  yolov8_bmcv.py   |      yolov8s_int8_4b.bmodel       |      3.95       |      4.18       |      11.83      |      6.89       |
|    SE9-8    |  yolov8_bmcv.soc  |      yolov8s_fp32_1b.bmodel       |      3.25       |      2.61       |     165.69      |      4.05       |
|    SE9-8    |  yolov8_bmcv.soc  |      yolov8s_fp16_1b.bmodel       |      3.27       |      2.60       |      37.13      |      4.02       |
|    SE9-8    |  yolov8_bmcv.soc  |      yolov8s_int8_1b.bmodel       |      3.27       |      2.60       |      9.54       |      4.04       |
|    SE9-8    |  yolov8_bmcv.soc  |      yolov8s_int8_4b.bmodel       |      3.07       |      2.48       |      9.55       |      3.99       |
|    SE9-8    |  yolov8_bmcv.py   |      yolov9s_fp32_1b.bmodel       |      4.16       |      4.51       |     169.02      |      7.59       |
|    SE9-8    |  yolov8_bmcv.py   |      yolov9s_fp16_1b.bmodel       |      4.17       |      4.50       |      47.11      |      7.56       |
|    SE9-8    |  yolov8_bmcv.py   |      yolov9s_int8_1b.bmodel       |      4.17       |      4.51       |      23.57      |      7.58       |
|    SE9-8    |  yolov8_bmcv.py   |      yolov9s_int8_4b.bmodel       |      3.95       |      4.17       |      22.70      |      6.80       |
|    SE9-8    |  yolov8_bmcv.soc  |      yolov9s_fp32_1b.bmodel       |      3.25       |      2.61       |     165.91      |      4.05       |
|    SE9-8    |  yolov8_bmcv.soc  |      yolov9s_fp16_1b.bmodel       |      3.27       |      2.60       |      44.02      |      4.03       |
|    SE9-8    |  yolov8_bmcv.soc  |      yolov9s_int8_1b.bmodel       |      3.27       |      2.60       |      20.50      |      4.03       |
|    SE9-8    |  yolov8_bmcv.soc  |      yolov9s_int8_4b.bmodel       |      3.07       |      2.48       |      20.37      |      4.01       |
|    SE9-8    |  yolov8_bmcv.py   |      yolov11s_fp32_1b.bmodel      |      4.21       |      4.52       |     138.64      |      7.56       |
|    SE9-8    |  yolov8_bmcv.py   |      yolov11s_fp16_1b.bmodel      |      4.16       |      4.49       |      39.61      |      7.55       |
|    SE9-8    |  yolov8_bmcv.py   |      yolov11s_int8_1b.bmodel      |      4.18       |      4.49       |      14.02      |      7.43       |
|    SE9-8    |  yolov8_bmcv.py   |      yolov11s_int8_4b.bmodel      |      3.96       |      4.18       |      12.79      |      6.65       |
|    SE9-8    |  yolov8_bmcv.soc  |      yolov11s_fp32_1b.bmodel      |      3.26       |      2.61       |     135.49      |      4.05       |
|    SE9-8    |  yolov8_bmcv.soc  |      yolov11s_fp16_1b.bmodel      |      3.27       |      2.60       |      36.59      |      4.04       |
|    SE9-8    |  yolov8_bmcv.soc  |      yolov11s_int8_1b.bmodel      |      3.27       |      2.60       |      10.93      |      4.03       |
|    SE9-8    |  yolov8_bmcv.soc  |      yolov11s_int8_4b.bmodel      |      3.08       |      2.48       |      10.47      |      4.00       |
|    SE9-8    |  yolov8_bmcv.py   |      yolov12s_fp32_1b.bmodel      |      8.81       |      4.52       |     213.03      |      7.63       |
|    SE9-8    |  yolov8_bmcv.soc  |      yolov12s_fp32_1b.bmodel      |      3.26       |      2.60       |     209.85      |      4.08       |
|    SE9-8    |  yolov8_bmcv.py   |      yolov12s_fp16_1b.bmodel      |      4.21       |      4.49       |      72.26      |      7.59       |
|    SE9-8    |  yolov8_bmcv.soc  |      yolov12s_fp16_1b.bmodel      |      3.24       |      2.60       |      69.05      |      4.05       |
|    SE9-8    |  yolov8_bmcv.py   |      yolov12s_int8_1b.bmodel      |      4.77       |      4.50       |      63.41      |      7.48       |
|    SE9-8    |  yolov8_bmcv.soc  |      yolov12s_int8_1b.bmodel      |      3.24       |      2.60       |      60.16      |      4.05       |
|    SE9-8    |  yolov8_bmcv.py   |      yolov12s_int8_4b.bmodel      |      4.48       |      4.15       |      62.67      |      6.63       |
|    SE9-8    |  yolov8_bmcv.soc  |      yolov12s_int8_4b.bmodel      |      3.05       |      2.49       |      60.36      |      4.02       |

> **测试说明**：  
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. SE5-16/SE7-32的主控处理器均为8核CA53@2.3GHz，SE9-16为8核CA53@1.6GHz，SE9-8为6核CA53@1.6GHzPCIe上的性能由于处理器的不同可能存在较大差异；
> 4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异，不同的阈值对后处理时间影响较大。 


## 8. FAQ
请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。
