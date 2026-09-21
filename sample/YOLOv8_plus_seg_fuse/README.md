# YOLOv8_plus_seg_fuse

## 目录

- [YOLOv8\_plus\_seg\_fuse](#yolov8_plus_seg_fuse)
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
  - [6. 性能测试](#6-性能测试)
    - [6.1 bmrt\_test](#61-bmrt_test)
    - [6.2 程序运行性能](#62-程序运行性能)
  - [7. FAQ](#7-faq)
  
## 1. 简介
YOLOv8_plus_seg_fuse例程将[YOLOv8_plus_seg例程](../YOLOv8_plus_seg/README.md)的部分前处理和后处理使用TPU来计算，大大提高了处理速度，目前支持使用BM1684X/BM1688/CV186X的INT8/FP16/FP32模型推理。

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
* 支持INT8/FP16/FP32模型编译和推理
* 支持C++、Python推理
* 支持图片和视频测试

## 3. 数据准备与模型编译

### 3.1 数据准备

本例程在`scripts`目录下提供了相关模型和数据的下载脚本`download.sh`，**如果您希望自己准备模型和数据集，可以跳过本小节，参考[3.2 模型编译](#32-模型编译)进行模型转换。**

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
│   ├── yolov8s_seg_fuse_fp32_1b.bmodel
│   ├── yolov8s_seg_fuse_fp16_1b.bmodel
│   └── yolov8s_seg_fuse_int8_1b.bmodel
├── BM1688 # 在BM1688上运行的模型
│   ├── yolov8s_seg_fuse_fp32_1b.bmodel
│   ├── yolov8s_seg_fuse_fp16_1b.bmodel
│   ├── yolov8s_seg_fuse_int8_1b.bmodel
│   ├── yolov8s_seg_fuse_fp32_1b_2core.bmodel
│   ├── yolov8s_seg_fuse_fp16_1b_2core.bmodel
│   └── yolov8s_seg_fuse_int8_1b_2core.bmodel
├── CV186X # 在CV186X上运行的模型
│   ├── yolov8s_seg_fuse_fp32_1b.bmodel
│   ├── yolov8s_seg_fuse_fp16_1b.bmodel
│   └── yolov8s_seg_fuse_int8_1b.bmodel
└── onnx
    └── yolov8s-seg.onnx
```
以上bmodel模型均由`yolov8s-seg.onnx`模型通过TPU-MLIR编译，bmodel模型名称中`1b`表示batch_size=1，`2core`表示num_core=2（适配BM1688）。
> **注：** 混合精度量化所需的qtable在`scripts`目录下（`yolov8s_seg_fuse_qtable`用于INT8，`yolov8s_seg_fuse_qtable_f16`用于FP16），随源码仓库一起提供，无需单独下载。
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

源模型需要编译成BModel才能在SOPHON TPU上运行，源模型在编译前要导出成onnx模型，如果您使用的TPU-MLIR版本>=v1.3.0（即官网v23.07.01），也可以直接使用torchscript模型。具体可参考[模型导出](./docs/YOLOv8_Export_Guide.md)。同时，您需要准备用于测试的数据集，如果量化模型，还要准备用于量化的数据集。

建议使用TPU-MLIR编译BModel，模型编译前需要安装TPU-MLIR，参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)中的pip下载方式来配置mlir环境。通过如下方式获取tpu-mlir的whl包：
```bash
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv8_plus_seg_fuse/tpu_mlir-1.30.2-py3-none-any.whl
```

安装好后需在TPU-MLIR环境中进入例程目录，并使用本例程提供的脚本将onnx模型编译为BModel。脚本中命令的详细说明可参考《TPU-MLIR开发手册》(请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的SDK中获取)。

- 生成FP32 BModel

本例程在`scripts`目录下提供了FP32 BModel的编译脚本`gen_fp32bmodel_mlir.sh`，执行时输入BModel的目标平台（**支持BM1684X/BM1688/CV186X**）：

```shell
./scripts/gen_fp32bmodel_mlir.sh bm1684x #bm1688 #cv186x
```

- 生成FP16 BModel

本例程在`scripts`目录下提供了FP16 BModel的编译脚本`gen_fp16bmodel_mlir.sh`，执行时输入BModel的目标平台（**支持BM1684X/BM1688/CV186X**）：

```shell
./scripts/gen_fp16bmodel_mlir.sh bm1684x #bm1688 #cv186x
```

> **注：** FP16 BModel同样使用了混合精度qtable（`scripts/yolov8s_seg_fuse_qtable_f16`），用于将box解码和融合后处理强制按F32精度计算。若不设置，BM1688上F16后处理的NMS去重会失效、多出约4倍框；且FP16的qtable与INT8的不同，只包含F32条目（F16编译下加入INT8条目会导致tpuc-opt报`CalibratedQuantizedType`断言错误）。

- 生成INT8 BModel

本例程在`scripts`目录下提供了量化INT8 BModel的脚本，请注意修改`gen_int8bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台（**支持BM1684X/BM1688/CV186X**），如：

```shell
./scripts/gen_int8bmodel_mlir.sh bm1684x #bm1688 #cv186x
```

上述脚本会在`models/BM1684X`等文件夹下生成转换好的INT8 BModel。

注：这里用到了混合精度量化，需要将一些层设为敏感层，相应的qtable在`scripts`目录下。如果您需要量化自己微调过的模型，可以参考[量化指南](../../docs/Calibration_Guide.md#13-特定模型优化技巧)中的方法，从我们提供的qtable倒推出自己模型需要的qtable。

## 4. 例程测试
- [C++例程](./cpp/README.md)
- [Python例程](./python/README.md)

## 5. 精度测试
### 5.1 测试方法

首先，参考[C++例程](cpp/README.md#32-测试图片)或[Python例程](python/README.md#22-测试图片)推理要测试的数据集，生成预测的json文件。  
然后，使用`tools`目录下的`eval_coco.py`脚本，将测试生成的json文件与测试集标签json文件进行对比，计算出实例分割的评价指标，命令如下：
```bash
# 安装pycocotools，若已安装请跳过
pip3 install pycocotools
# 请根据实际情况修改程序路径和json文件路径
python3 tools/eval_coco.py --gt_path datasets/coco/instances_val2017_1000.json --result_json results/yolov8s_seg_fuse_int8_1b.bmodel_val2017_1000_bmcv_python_result.json --ann_type=segm
```

> **注意：** seg_fuse模型的置信度阈值和NMS阈值已内置在TPU后处理中，Python和C++例程无需额外的`--conf_thresh`/`--nms_thresh`参数。

### 5.2 测试结果
在coco2017 val数据集上（1000张样本），精度测试结果如下：

|   测试平台    |      测试程序     |      测试模型          |AP@IoU=0.5:0.95|AP@IoU=0.5|
| ------------ | ---------------- | ---------------------- | ------------- | -------- |
|   SE7-32     |  yolov8_bmcv.py   |  yolov8s_seg_fuse_fp32_1b.bmodel  | 0.327 | 0.506 |
|   SE7-32     |  yolov8_bmcv.soc  |  yolov8s_seg_fuse_fp32_1b.bmodel  | 0.328 | 0.506 |
|   SE7-32     |  yolov8_bmcv.py   |  yolov8s_seg_fuse_fp16_1b.bmodel  | 0.327 | 0.506 |
|   SE7-32     |  yolov8_bmcv.soc  |  yolov8s_seg_fuse_fp16_1b.bmodel  | 0.328 | 0.506 |
|   SE7-32     |  yolov8_bmcv.py   |  yolov8s_seg_fuse_int8_1b.bmodel  | 0.323 | 0.496 |
|   SE7-32     |  yolov8_bmcv.soc  |  yolov8s_seg_fuse_int8_1b.bmodel  | 0.322 | 0.495 |
|   SE9-16     |  yolov8_bmcv.py   |  yolov8s_seg_fuse_fp32_1b.bmodel  | 0.327 | 0.506 |
|   SE9-16     |  yolov8_bmcv.soc  |  yolov8s_seg_fuse_fp32_1b.bmodel  | 0.328 | 0.506 |
|   SE9-16     |  yolov8_bmcv.py   |  yolov8s_seg_fuse_fp16_1b.bmodel  | 0.327 | 0.506 |
|   SE9-16     |  yolov8_bmcv.soc  |  yolov8s_seg_fuse_fp16_1b.bmodel  | 0.328 | 0.506 |
|   SE9-16     |  yolov8_bmcv.py   |  yolov8s_seg_fuse_int8_1b.bmodel  | 0.325 | 0.497 |
|   SE9-16     |  yolov8_bmcv.soc  |  yolov8s_seg_fuse_int8_1b.bmodel  | 0.324 | 0.497 |
|   SE9-16     |  yolov8_bmcv.py   |  yolov8s_seg_fuse_fp32_1b_2core.bmodel  | 0.327 | 0.506 |
|   SE9-16     |  yolov8_bmcv.soc  |  yolov8s_seg_fuse_fp32_1b_2core.bmodel  | 0.328 | 0.506 |
|   SE9-16     |  yolov8_bmcv.py   |  yolov8s_seg_fuse_fp16_1b_2core.bmodel  | 0.327 | 0.506 |
|   SE9-16     |  yolov8_bmcv.soc  |  yolov8s_seg_fuse_fp16_1b_2core.bmodel  | 0.328 | 0.506 |
|   SE9-16     |  yolov8_bmcv.py   |  yolov8s_seg_fuse_int8_1b_2core.bmodel  | 0.325 | 0.497 |
|   SE9-16     |  yolov8_bmcv.soc  |  yolov8s_seg_fuse_int8_1b_2core.bmodel  | 0.324 | 0.497 |
|   SE9-8      |  yolov8_bmcv.py   |  yolov8s_seg_fuse_fp32_1b.bmodel  | 0.327 | 0.506 |
|   SE9-8      |  yolov8_bmcv.soc  |  yolov8s_seg_fuse_fp32_1b.bmodel  | 0.328 | 0.506 |
|   SE9-8      |  yolov8_bmcv.py   |  yolov8s_seg_fuse_fp16_1b.bmodel  | 0.327 | 0.506 |
|   SE9-8      |  yolov8_bmcv.soc  |  yolov8s_seg_fuse_fp16_1b.bmodel  | 0.328 | 0.506 |
|   SE9-8      |  yolov8_bmcv.py   |  yolov8s_seg_fuse_int8_1b.bmodel  | 0.325 | 0.497 |
|   SE9-8      |  yolov8_bmcv.soc  |  yolov8s_seg_fuse_int8_1b.bmodel  | 0.324 | 0.497 |

> **测试说明**：  
> 1. 由于seg_fuse例程将部分后处理（包括置信度过滤和NMS）融合到TPU中计算，阈值由模型编译时确定，与[YOLOv8_plus_seg例程](../YOLOv8_plus_seg/README.md)使用外部阈值(conf_thresh=0.001, nms_thresh=0.7)的测试方式不同，因此AP值会有一定差异；
> 2. AP@IoU=0.5:0.95为area=all对应的指标；
> 3. SE7-32与SE9-16的精度指标由于平台差异（libsophon版本、编译器版本等）有一定浮动；
> 4. Python和C++例程的精度基本一致，差异来自于后处理实现的微小不同；
> 5. INT8模型采用混合精度量化（检测head为F16、box解码与融合后处理为F32），以修复BM1688上INT8 head + F16后处理导致的NMS无效化问题，使SE9-16的INT8精度达到与SE7-32一致的水平；
> 6. BM1688的2-core模型（`*_1b_2core.bmodel`）与对应1-core模型精度一致，可在无精度损失的情况下获得推理提速（见[6.2 程序运行性能](#62-程序运行性能)）。
> 7. SE9-8对应CV186X芯片（与BM1688为同一TPU），其精度与SE9-16基本一致。
> 8. Python例程将mask裁剪所用的bbox ROI取整由`ceil`改为与C++一致的`std::round`（见`python/postprocess_bmcv.py`），mask按该ROI裁剪，因此Python的AP较旧的ceil实现下降约0.003，但与C++例程的差距亦由0.002~0.003缩小至约0.001。

## 6. 性能测试
### 6.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684X/yolov8s_seg_fuse_fp32_1b.bmodel
# BM1688平台请使用models/BM1688/下的bmodel，CV186X平台请使用models/CV186X/下的bmodel，2-core模型请使用*_1b_2core.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

|    测试平台  |              测试模型           | calculate time(ms) |
| ----------- | -------------------------------| ----------------- |
|   SE7-32    | BM1684X/yolov8s_seg_fuse_fp32_1b.bmodel  |         45.65  |
|   SE7-32    | BM1684X/yolov8s_seg_fuse_fp16_1b.bmodel  |         11.11  |
|   SE7-32    | BM1684X/yolov8s_seg_fuse_int8_1b.bmodel  |          8.92  |
|   SE9-16    | BM1688/yolov8s_seg_fuse_fp32_1b.bmodel   |        240.06  |
|   SE9-16    | BM1688/yolov8s_seg_fuse_fp16_1b.bmodel   |         51.78  |
|   SE9-16    | BM1688/yolov8s_seg_fuse_int8_1b.bmodel   |         29.57  |
|   SE9-16    | BM1688/yolov8s_seg_fuse_fp32_1b_2core.bmodel |     130.74  |
|   SE9-16    | BM1688/yolov8s_seg_fuse_fp16_1b_2core.bmodel |      33.10  |
|   SE9-16    | BM1688/yolov8s_seg_fuse_int8_1b_2core.bmodel |      21.56  |
|   SE9-8     | CV186X/yolov8s_seg_fuse_fp32_1b.bmodel   |        239.64  |
|   SE9-8     | CV186X/yolov8s_seg_fuse_fp16_1b.bmodel   |         51.47  |
|   SE9-8     | CV186X/yolov8s_seg_fuse_int8_1b.bmodel   |         29.44  |

### 6.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++和Python例程打印的时间已经折算为单张图片的处理时间。

在不同的测试平台上，使用不同的例程、模型测试`datasets/coco/val2017_1000`，性能测试结果如下：
|    测试平台  |     测试程序      |        测试模型        |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ---------------- | ---------------------- | -------- | --------- | --------- | --------- |
|   SE7-32    |  yolov8_bmcv.py   |  yolov8s_seg_fuse_fp32_1b.bmodel     |      3.04       |      1.17       |      44.73      |      9.52       |
|   SE7-32    |  yolov8_bmcv.soc  |  yolov8s_seg_fuse_fp32_1b.bmodel     |      2.70       |      0.45       |      44.51      |      6.50       |
|   SE7-32    |  yolov8_bmcv.py   |  yolov8s_seg_fuse_fp16_1b.bmodel     |      3.04       |      1.17       |      9.76       |      10.60      |
|   SE7-32    |  yolov8_bmcv.soc  |  yolov8s_seg_fuse_fp16_1b.bmodel     |      2.69       |      0.45       |      9.55       |      6.63       |
|   SE7-32    |  yolov8_bmcv.py   |  yolov8s_seg_fuse_int8_1b.bmodel     |      3.01       |      1.17       |      7.04       |      7.46       |
|   SE7-32    |  yolov8_bmcv.soc  |  yolov8s_seg_fuse_int8_1b.bmodel     |      2.68       |      0.45       |      6.84       |      2.78       |
|   SE9-16    |  yolov8_bmcv.py   |  yolov8s_seg_fuse_fp32_1b.bmodel  |      3.99       |      2.80       |      239.68     |      12.40      |
|   SE9-16    |  yolov8_bmcv.soc  |  yolov8s_seg_fuse_fp32_1b.bmodel  |      3.40       |      1.20       |      239.26     |      8.73       |
|   SE9-16    |  yolov8_bmcv.py   |  yolov8s_seg_fuse_fp16_1b.bmodel  |      3.97       |      2.80       |      50.64      |      13.17      |
|   SE9-16    |  yolov8_bmcv.soc  |  yolov8s_seg_fuse_fp16_1b.bmodel  |      3.41       |      1.19       |      50.29      |      9.05       |
|   SE9-16    |  yolov8_bmcv.py   |  yolov8s_seg_fuse_int8_1b.bmodel  |      3.96       |      2.80       |      29.09      |      12.91      |
|   SE9-16    |  yolov8_bmcv.soc  |  yolov8s_seg_fuse_int8_1b.bmodel  |      3.35       |      1.20       |      28.75      |      6.56       |
|   SE9-16    |  yolov8_bmcv.py   |  yolov8s_seg_fuse_fp32_1b_2core.bmodel  |      3.97       |      2.80       |      129.88     |      12.40      |
|   SE9-16    |  yolov8_bmcv.soc  |  yolov8s_seg_fuse_fp32_1b_2core.bmodel  |      3.40       |      1.20       |      129.50     |      9.04       |
|   SE9-16    |  yolov8_bmcv.py   |  yolov8s_seg_fuse_fp16_1b_2core.bmodel  |      3.97       |      2.80       |      32.14      |      13.17      |
|   SE9-16    |  yolov8_bmcv.soc  |  yolov8s_seg_fuse_fp16_1b_2core.bmodel  |      3.39       |      1.19       |      31.79      |      9.00       |
|   SE9-16    |  yolov8_bmcv.py   |  yolov8s_seg_fuse_int8_1b_2core.bmodel  |      3.95       |      2.80       |      21.25      |      12.91      |
|   SE9-16    |  yolov8_bmcv.soc  |  yolov8s_seg_fuse_int8_1b_2core.bmodel  |      3.37       |      1.19       |      20.91      |      6.55       |
|   SE9-8     |  yolov8_bmcv.py   |  yolov8s_seg_fuse_fp32_1b.bmodel  |      3.86       |      2.82       |      239.71     |      13.38      |
|   SE9-8     |  yolov8_bmcv.soc  |  yolov8s_seg_fuse_fp32_1b.bmodel  |      4.35       |      1.20       |      239.26     |      10.16      |
|   SE9-8     |  yolov8_bmcv.py   |  yolov8s_seg_fuse_fp16_1b.bmodel  |      3.87       |      2.81       |      50.64      |      14.55      |
|   SE9-8     |  yolov8_bmcv.soc  |  yolov8s_seg_fuse_fp16_1b.bmodel  |      3.39       |      1.20       |      50.27      |      9.74       |
|   SE9-8     |  yolov8_bmcv.py   |  yolov8s_seg_fuse_int8_1b.bmodel  |      3.85       |      2.81       |      29.09      |      13.52      |
|   SE9-8     |  yolov8_bmcv.soc  |  yolov8s_seg_fuse_int8_1b.bmodel  |      3.33       |      1.20       |      28.72      |      6.43       |

> **测试说明**：  
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. SE5/SE7-32的主控处理器均为8核CA53@2.3GHz，SE9-16为8核CA53@1.6GHz，SE9-8为6核CA53@1.65GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异；
> 5. seg_fuse例程将box解码和实例分割后处理融合到TPU中且以F32精度计算，因此BM1688上FP32/FP16模型的inference_time明显高于INT8模型；FP32/FP16模型的mask输出为浮点类型（走opencv resize），INT8模型的mask输出为UINT8（走BMCV resize），后处理仅对通过置信度过滤的实例做resize并按bbox ROI裁剪mask，三个精度的Python例程postprocess_time均已降至约13ms；
> 6. BM1688额外提供2-core模型（`yolov8s_seg_fuse_*_1b_2core.bmodel`），其精度与1-core模型一致，inference_time约为1-core的54%~73%（fp32 1.85×、fp16 1.58×、int8 1.37×提速）。2-core模型依赖融合后处理（`tpu.yolo_seg_post`动态输出）的2-core编译修复，请使用新版TPU-MLIR（v1.30.2-20260917及之后）编译，旧版TPU-MLIR会生成错误的动态输出。
> 7. Python后处理已优化：findContours/轮廓生成（仅用于可视化绘制）移出postprocess计时；postprocess只保留mask缩放、ROI裁剪与阈值比较，因此Python例程postprocess_time由优化前的约27~110ms降至约13ms。


## 7. FAQ
请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。
