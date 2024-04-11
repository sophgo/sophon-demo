# YOLOv8

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
​YOLOv8是YOLO系列的的一个重大更新版本，它抛弃了以往的YOLO系类模型使用的Anchor-Base，采用了Anchor-Free的思想。YOLOv8建立在YOLO系列成功的基础上，通过对网络结构的改造，进一步提升其性能和灵活性。本例程对[​YOLOv8官方开源仓库](https://github.com/ultralytics/ultralytics)的模型和算法进行移植，使之能在SOPHON BM1684/BM1684X/BM1688/CV186X上进行推理测试。

## 2. 特性
* 支持BM1688/CV186X(SoC)、BM1684X(x86 PCIe、SoC)、BM1684(x86 PCIe、SoC、arm PCIe)
* 支持FP32、FP16(BM1684X/BM1688/CV186X)、INT8模型编译和推理
* 支持基于BMCV预处理的C++推理
* 支持基于OpenCV和BMCV预处理的Python推理
* 支持单batch和多batch模型推理
* 支持1个输出模型推理
* C++例程支持opt模型推理（Python不支持）
* 支持图片和视频测试

## 3. 准备模型与数据
建议使用TPU-MLIR编译BModel，在使用TPU-MLIR编译前需要导出ONNX模型。具体可参考[YOLOv8模型导出](./docs/YOLOv8_Export_Guide.md)。

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
│   ├── yolov8s_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684的FP32 BModel，batch_size=1
│   ├── yolov8s_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684的INT8 BModel，batch_size=1
│   ├── yolov8s_int8_4b.bmodel   # 使用TPU-MLIR编译，用于BM1684的INT8 BModel，batch_size=4
│   ├── yolov8s_opt_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684的FP32 BModel，batch_size=1
│   ├── yolov8s_opt_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684的INT8 BModel，batch_size=1
│   └── yolov8s_opt_int8_4b.bmodel   # 使用TPU-MLIR编译，用于BM1684的INT8 BModel，batch_size=4
├── BM1684X
│   ├── yolov8s_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=1
│   ├── yolov8s_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=1
│   ├── yolov8s_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=1
│   ├── yolov8s_int8_4b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=4
│   ├── yolov8s_opt_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=1
│   ├── yolov8s_opt_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=1
│   ├── yolov8s_opt_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=1
│   └── yolov8s_opt_int8_4b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=4
├── BM1688
│   ├── yolov8s_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1, num_core=1
│   ├── yolov8s_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1, num_core=1
│   ├── yolov8s_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=1, num_core=1
│   ├── yolov8s_int8_4b.bmodel   # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=4, num_core=1
│   ├── yolov8s_fp32_1b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1, num_core=2
│   ├── yolov8s_fp16_1b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1, num_core=2
│   ├── yolov8s_int8_1b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=1, num_core=2
│   ├── yolov8s_int8_4b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=4, num_core=2
│   ├── yolov8s_opt_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1, num_core=1
│   ├── yolov8s_opt_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1, num_core=1
│   ├── yolov8s_opt_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=1, num_core=1
│   ├── yolov8s_opt_int8_4b.bmodel   # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=4, num_core=1
│   ├── yolov8s_opt_fp32_1b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1, num_core=2
│   ├── yolov8s_opt_fp16_1b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1, num_core=2
│   ├── yolov8s_opt_int8_1b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=1, num_core=2
│   └── yolov8s_opt_int8_4b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=4, num_core=2
├── CV186X
│   ├── yolov8s_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于CV186X的FP32 BModel，batch_size=1
│   ├── yolov8s_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于CV186X的FP16 BModel，batch_size=1
│   ├── yolov8s_int8_1b.bmodel   # 使用TPU-MLIR编译，用于CV186X的INT8 BModel，batch_size=1
│   ├── yolov8s_int8_4b.bmodel   # 使用TPU-MLIR编译，用于CV186X的INT8 BModel，batch_size=4
│   ├── yolov8s_opt_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于CV186X的FP32 BModel，batch_size=1
│   ├── yolov8s_opt_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于CV186X的FP16 BModel，batch_size=1
│   ├── yolov8s_opt_int8_1b.bmodel   # 使用TPU-MLIR编译，用于CV186X的INT8 BModel，batch_size=1
│   └── yolov8s_opt_int8_4b.bmodel   # 使用TPU-MLIR编译，用于CV186X的INT8 BModel，batch_size=4
│── torch
│   └── yolov8s.torchscript.pt   # trace后的torchscript模型
└── onnx
    ├── yolov8s.onnx      # 导出的动态onnx模型
    ├── yolov8s_opt.onnx      # 导出的动态opt onnx模型
    ├── yolov8s_qtable_fp16       # TPU-MLIR编译时，用于BM1684X/BM1688的INT8 BModel混合精度量化
    ├── yolov8s_qtable_fp32       # TPU-MLIR编译时，用于BM1684的INT8 BModel混合精度量化
    ├── yolov8s_opt_qtable_fp16       # TPU-MLIR编译时，用于BM1684X/BM1688的INT8 BModel混合精度量化
    └── yolov8s_opt_qtable_fp32       # TPU-MLIR编译时，用于BM1684的INT8 BModel混合精度量化
    
    
         
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

​执行上述命令会在`models/BM1684`下生成`yolov8s_fp32_1b.bmodel`和`yolov8s_opt_fp32_1b.bmodel`文件，即转换好的FP32 BModel。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x #bm1688/cv186x
```

​执行上述命令会在`models/BM1684X/`下生成`yolov8s_fp16_1b.bmodel`和`yolov8s_opt_fp16_1b.bmodel`文件，即转换好的FP16 BModel。

- 生成INT8 BModel

​本例程在`scripts`目录下提供了量化INT8 BModel的脚本，请注意修改`gen_int8bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台（**支持BM1684/BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_int8bmodel_mlir.sh bm1684 #bm1684x/bm1688/cv186x
```

​执行上述命令会在`models/BM1684`下生成`yolov8s_int8_1b.bmodel`和`yolov8s_opt_int8_1b.bmodel`等文件，即转换好的INT8 BModel。量化模型出现问题可以参考：[Calibration_Guide](../../docs/Calibration_Guide.md)。


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
| SE5-16       | yolov8_opencv.py | yolov8s_fp32_1b.bmodel |    0.448 |    0.609 |
| SE5-16       | yolov8_opencv.py | yolov8s_int8_1b.bmodel |    0.435 |    0.596 |
| SE5-16       | yolov8_opencv.py | yolov8s_int8_4b.bmodel |    0.435 |    0.596 |
| SE5-16       | yolov8_bmcv.py | yolov8s_fp32_1b.bmodel |    0.448 |    0.610 |
| SE5-16       | yolov8_bmcv.py | yolov8s_int8_1b.bmodel |    0.435 |    0.597 |
| SE5-16       | yolov8_bmcv.py | yolov8s_int8_4b.bmodel |    0.435 |    0.597 |
| SE5-16       | yolov8_bmcv.soc | yolov8s_fp32_1b.bmodel |    0.448 |    0.609 |
| SE5-16       | yolov8_bmcv.soc | yolov8s_int8_1b.bmodel |    0.435 |    0.596 |
| SE5-16       | yolov8_bmcv.soc | yolov8s_int8_4b.bmodel |    0.435 |    0.596 |
| SE5-16       | yolov8_bmcv.soc | yolov8s_opt_fp32_1b.bmodel |    0.448 |    0.609 |
| SE5-16       | yolov8_bmcv.soc | yolov8s_opt_int8_1b.bmodel |    0.425 |    0.587 |
| SE5-16       | yolov8_bmcv.soc | yolov8s_opt_int8_4b.bmodel |    0.425 |    0.587 |
| SE7-32       | yolov8_opencv.py | yolov8s_fp32_1b.bmodel |    0.448 |    0.609 |
| SE7-32       | yolov8_opencv.py | yolov8s_fp16_1b.bmodel |    0.447 |    0.609 |
| SE7-32       | yolov8_opencv.py | yolov8s_int8_1b.bmodel |    0.443 |    0.606 |
| SE7-32       | yolov8_opencv.py | yolov8s_int8_4b.bmodel |    0.443 |    0.606 |
| SE7-32       | yolov8_bmcv.py | yolov8s_fp32_1b.bmodel |    0.447 |    0.610 |
| SE7-32       | yolov8_bmcv.py | yolov8s_fp16_1b.bmodel |    0.447 |    0.610 |
| SE7-32       | yolov8_bmcv.py | yolov8s_int8_1b.bmodel |    0.442 |    0.606 |
| SE7-32       | yolov8_bmcv.py | yolov8s_int8_4b.bmodel |    0.442 |    0.606 |
| SE7-32       | yolov8_bmcv.soc | yolov8s_fp32_1b.bmodel |    0.448 |    0.610 |
| SE7-32       | yolov8_bmcv.soc | yolov8s_fp16_1b.bmodel |    0.447 |    0.609 |
| SE7-32       | yolov8_bmcv.soc | yolov8s_int8_1b.bmodel |    0.443 |    0.606 |
| SE7-32       | yolov8_bmcv.soc | yolov8s_int8_4b.bmodel |    0.443 |    0.606 |
| SE7-32       | yolov8_bmcv.soc | yolov8s_opt_fp32_1b.bmodel |    0.448 |    0.610 |
| SE7-32       | yolov8_bmcv.soc | yolov8s_opt_fp16_1b.bmodel |    0.447 |    0.609 |
| SE7-32       | yolov8_bmcv.soc | yolov8s_opt_int8_1b.bmodel |    0.442 |    0.605 |
| SE7-32       | yolov8_bmcv.soc | yolov8s_opt_int8_4b.bmodel |    0.442 |    0.605 |
| SE9-16       | yolov8_opencv.py | yolov8s_fp32_1b.bmodel |    0.448 |    0.609 |
| SE9-16       | yolov8_opencv.py | yolov8s_fp16_1b.bmodel |    0.447 |    0.609 |
| SE9-16       | yolov8_opencv.py | yolov8s_int8_1b.bmodel |    0.443 |    0.607 |
| SE9-16       | yolov8_opencv.py | yolov8s_int8_4b.bmodel |    0.443 |    0.607 |
| SE9-16       | yolov8_bmcv.py | yolov8s_fp32_1b.bmodel |    0.447 |    0.610 |
| SE9-16       | yolov8_bmcv.py | yolov8s_fp16_1b.bmodel |    0.447 |    0.610 |
| SE9-16       | yolov8_bmcv.py | yolov8s_int8_1b.bmodel |    0.443 |    0.607 |
| SE9-16       | yolov8_bmcv.py | yolov8s_int8_4b.bmodel |    0.443 |    0.607 |
| SE9-16       | yolov8_bmcv.soc | yolov8s_fp32_1b.bmodel |    0.448 |    0.610 |
| SE9-16       | yolov8_bmcv.soc | yolov8s_fp16_1b.bmodel |    0.448 |    0.609 |
| SE9-16       | yolov8_bmcv.soc | yolov8s_int8_1b.bmodel |    0.443 |    0.607 |
| SE9-16       | yolov8_bmcv.soc | yolov8s_int8_4b.bmodel |    0.443 |    0.607 |
| SE9-16       | yolov8_opencv.py | yolov8s_fp32_1b_2core.bmodel |    0.448 |    0.609 |
| SE9-16       | yolov8_opencv.py | yolov8s_fp16_1b_2core.bmodel |    0.447 |    0.609 |
| SE9-16       | yolov8_opencv.py | yolov8s_int8_1b_2core.bmodel |    0.443 |    0.607 |
| SE9-16       | yolov8_opencv.py | yolov8s_int8_4b_2core.bmodel |    0.443 |    0.607 |
| SE9-16       | yolov8_bmcv.py | yolov8s_fp32_1b_2core.bmodel |    0.447 |    0.610 |
| SE9-16       | yolov8_bmcv.py | yolov8s_fp16_1b_2core.bmodel |    0.447 |    0.610 |
| SE9-16       | yolov8_bmcv.py | yolov8s_int8_1b_2core.bmodel |    0.443 |    0.607 |
| SE9-16       | yolov8_bmcv.py | yolov8s_int8_4b_2core.bmodel |    0.443 |    0.607 |
| SE9-16       | yolov8_bmcv.soc | yolov8s_fp32_1b_2core.bmodel |    0.448 |    0.610 |
| SE9-16       | yolov8_bmcv.soc | yolov8s_fp16_1b_2core.bmodel |    0.448 |    0.609 |
| SE9-16       | yolov8_bmcv.soc | yolov8s_int8_1b_2core.bmodel |    0.443 |    0.607 |
| SE9-16       | yolov8_bmcv.soc | yolov8s_int8_4b_2core.bmodel |    0.443 |    0.607 |
| SE9-16       | yolov8_bmcv.soc | yolov8s_opt_fp32_1b.bmodel |    0.448 |    0.610 |
| SE9-16       | yolov8_bmcv.soc | yolov8s_opt_fp16_1b.bmodel |    0.447 |    0.609 |
| SE9-16       | yolov8_bmcv.soc | yolov8s_opt_int8_1b.bmodel |    0.441 |    0.605 |
| SE9-16       | yolov8_bmcv.soc | yolov8s_opt_int8_4b.bmodel |    0.441 |    0.605 |
| SE9-16       | yolov8_bmcv.soc | yolov8s_opt_fp32_1b_2core.bmodel |    0.448 |    0.610 |
| SE9-16       | yolov8_bmcv.soc | yolov8s_opt_fp16_1b_2core.bmodel |    0.447 |    0.609 |
| SE9-16       | yolov8_bmcv.soc | yolov8s_opt_int8_1b_2core.bmodel |    0.441 |    0.605 |
| SE9-16       | yolov8_bmcv.soc | yolov8s_opt_int8_4b_2core.bmodel |    0.441 |    0.605 |
| SE9-8        | yolov8_opencv.py | yolov8s_fp32_1b.bmodel |    0.448 |    0.609 |
| SE9-8        | yolov8_opencv.py | yolov8s_fp16_1b.bmodel |    0.448 |    0.609 |
| SE9-8        | yolov8_opencv.py | yolov8s_int8_1b.bmodel |    0.444 |    0.606 |
| SE9-8        | yolov8_opencv.py | yolov8s_int8_4b.bmodel |    0.444 |    0.606 |
| SE9-8        | yolov8_bmcv.py | yolov8s_fp32_1b.bmodel |    0.447 |    0.610 |
| SE9-8        | yolov8_bmcv.py | yolov8s_fp16_1b.bmodel |    0.447 |    0.610 |
| SE9-8        | yolov8_bmcv.py | yolov8s_int8_1b.bmodel |    0.443 |    0.607 |
| SE9-8        | yolov8_bmcv.py | yolov8s_int8_4b.bmodel |    0.443 |    0.607 |
| SE9-8        | yolov8_bmcv.soc | yolov8s_fp32_1b.bmodel |    0.448 |    0.610 |
| SE9-8        | yolov8_bmcv.soc | yolov8s_fp16_1b.bmodel |    0.448 |    0.609 |
| SE9-8        | yolov8_bmcv.soc | yolov8s_int8_1b.bmodel |    0.443 |    0.607 |
| SE9-8        | yolov8_bmcv.soc | yolov8s_int8_4b.bmodel |    0.443 |    0.607 |
| SE9-8        | yolov8_bmcv.soc | yolov8s_opt_fp32_1b.bmodel |    0.448 |    0.610 |
| SE9-8        | yolov8_bmcv.soc | yolov8s_opt_fp16_1b.bmodel |    0.447 |    0.609 |
| SE9-8        | yolov8_bmcv.soc | yolov8s_opt_int8_1b.bmodel |    0.442 |    0.606 |
| SE9-8        | yolov8_bmcv.soc | yolov8s_opt_int8_4b.bmodel |    0.442 |    0.606 |
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
bmrt_test --bmodel models/BM1684/yolov8s_fp32_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

|              测试模型               | calculate time(ms) |
| ----------------------------------- | ----------------- |
| BM1684/yolov8s_fp32_1b.bmodel      |          26.47  |
| BM1684/yolov8s_int8_1b.bmodel      |          15.65  |
| BM1684/yolov8s_int8_4b.bmodel      |           9.36  |
| BM1684/yolov8s_opt_fp32_1b.bmodel  |          26.71  |
| BM1684/yolov8s_opt_int8_1b.bmodel  |          15.09  |
| BM1684/yolov8s_opt_int8_4b.bmodel  |           7.39  |
| BM1684X/yolov8s_fp32_1b.bmodel     |          29.20  |
| BM1684X/yolov8s_fp16_1b.bmodel     |           5.48  |
| BM1684X/yolov8s_int8_1b.bmodel     |           3.32  |
| BM1684X/yolov8s_int8_4b.bmodel     |           3.33  |
| BM1684X/yolov8s_opt_fp32_1b.bmodel |          29.40  |
| BM1684X/yolov8s_opt_fp16_1b.bmodel |           5.62  |
| BM1684X/yolov8s_opt_int8_1b.bmodel |           2.94  |
| BM1684X/yolov8s_opt_int8_4b.bmodel |           2.91  |
| BM1688/yolov8s_fp32_1b.bmodel      |         165.56  |
| BM1688/yolov8s_fp16_1b.bmodel      |          36.27  |
| BM1688/yolov8s_int8_1b.bmodel      |          12.62  |
| BM1688/yolov8s_int8_4b.bmodel      |          11.99  |
| BM1688/yolov8s_fp32_1b_2core.bmodel|          96.41  |
| BM1688/yolov8s_fp16_1b_2core.bmodel|          24.29  |
| BM1688/yolov8s_int8_1b_2core.bmodel|          10.14  |
| BM1688/yolov8s_int8_4b_2core.bmodel|           7.68  |
| BM1688/yolov8s_opt_fp32_1b.bmodel  |         165.22  |
| BM1688/yolov8s_opt_fp16_1b.bmodel  |          36.18  |
| BM1688/yolov8s_opt_int8_1b.bmodel  |           9.89  |
| BM1688/yolov8s_opt_int8_4b.bmodel  |           9.27  |
| BM1688/yolov8s_opt_fp32_1b_2core.bmodel|          96.67  |
| BM1688/yolov8s_opt_fp16_1b_2core.bmodel|          24.43  |
| BM1688/yolov8s_opt_int8_1b_2core.bmodel|           9.01  |
| BM1688/yolov8s_opt_int8_4b_2core.bmodel|           6.25  |
| CV186X/yolov8s_fp32_1b.bmodel      |         161.60  |
| CV186X/yolov8s_fp16_1b.bmodel      |          34.18  |
| CV186X/yolov8s_int8_1b.bmodel      |          10.84  |
| CV186X/yolov8s_int8_4b.bmodel      |          10.49  |
| CV186X/yolov8s_opt_fp32_1b.bmodel  |         161.89  |
| CV186X/yolov8s_opt_fp16_1b.bmodel  |          34.27  |
| CV186X/yolov8s_opt_int8_1b.bmodel  |           7.90  |
| CV186X/yolov8s_opt_int8_4b.bmodel  |           7.57  |
> **测试说明**：  
1. 性能测试结果具有一定的波动性；
2. `calculate time`已折算为平均每张图片的推理时间；
3. SoC和PCIe的测试结果基本一致。


### 7.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++和Python例程打印的时间已经折算为单张图片的处理时间。

在不同的测试平台上，使用不同的例程、模型测试`datasets/val2017_1000`，conf_thresh=0.25，nms_thresh=0.7，性能测试结果如下：
|    测试平台  |     测试程序      |        测试模型        |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ---------------- | ---------------------- | --------  | ---------    | ---------     | ---------      |
|   SE5-16    | yolov8_opencv.py  | yolov8s_fp32_1b.bmodel  |      18.98      |      21.50      |      31.77      |      5.11       |
|   SE5-16    | yolov8_opencv.py  | yolov8s_int8_1b.bmodel  |      15.02      |      21.35      |      20.80      |      4.85       |
|   SE5-16    | yolov8_opencv.py  | yolov8s_int8_4b.bmodel  |      15.06      |      23.84      |      14.39      |      5.15       |
|   SE5-16    |  yolov8_bmcv.py   | yolov8s_fp32_1b.bmodel  |      3.63       |      2.80       |      28.88      |      4.99       |
|   SE5-16    |  yolov8_bmcv.py   | yolov8s_int8_1b.bmodel  |      3.67       |      2.83       |      18.08      |      4.97       |
|   SE5-16    |  yolov8_bmcv.py   | yolov8s_int8_4b.bmodel  |      3.51       |      2.67       |      11.33      |      4.37       |
|   SE5-16    |  yolov8_bmcv.soc  | yolov8s_fp32_1b.bmodel  |      4.86       |      1.54       |      26.41      |      8.53       |
|   SE5-16    |  yolov8_bmcv.soc  | yolov8s_int8_1b.bmodel  |      4.89       |      1.55       |      15.58      |      8.53       |
|   SE5-16    |  yolov8_bmcv.soc  | yolov8s_int8_4b.bmodel  |      4.75       |      1.49       |      9.34       |      8.50       |
|   SE5-16    |  yolov8_bmcv.soc  |yolov8s_opt_fp32_1b.bmodel|      4.94       |      1.54       |      26.62      |      2.65       |
|   SE5-16    |  yolov8_bmcv.soc  |yolov8s_opt_int8_1b.bmodel|      4.88       |      1.54       |      15.02      |      2.60       |
|   SE5-16    |  yolov8_bmcv.soc  |yolov8s_opt_int8_4b.bmodel|      4.82       |      1.49       |      7.37       |      2.66       |
|   SE7-32    | yolov8_opencv.py  | yolov8s_fp32_1b.bmodel  |      21.70      |      22.55      |      35.12      |      5.38       |
|   SE7-32    | yolov8_opencv.py  | yolov8s_fp16_1b.bmodel  |      14.95      |      22.53      |      11.42      |      5.39       |
|   SE7-32    | yolov8_opencv.py  | yolov8s_int8_1b.bmodel  |      14.96      |      22.87      |      9.21       |      5.36       |
|   SE7-32    | yolov8_opencv.py  | yolov8s_int8_4b.bmodel  |      15.00      |      24.62      |      8.98       |      5.45       |
|   SE7-32    |  yolov8_bmcv.py   | yolov8s_fp32_1b.bmodel  |      3.16       |      2.38       |      31.87      |      5.42       |
|   SE7-32    |  yolov8_bmcv.py   | yolov8s_fp16_1b.bmodel  |      3.16       |      2.37       |      8.13       |      5.45       |
|   SE7-32    |  yolov8_bmcv.py   | yolov8s_int8_1b.bmodel  |      3.16       |      2.37       |      5.96       |      5.43       |
|   SE7-32    |  yolov8_bmcv.py   | yolov8s_int8_4b.bmodel  |      2.97       |      2.17       |      5.52       |      4.87       |
|   SE7-32    |  yolov8_bmcv.soc  | yolov8s_fp32_1b.bmodel  |      4.36       |      0.74       |      29.15      |      8.65       |
|   SE7-32    |  yolov8_bmcv.soc  | yolov8s_fp16_1b.bmodel  |      4.39       |      0.74       |      5.46       |      8.67       |
|   SE7-32    |  yolov8_bmcv.soc  | yolov8s_int8_1b.bmodel  |      4.38       |      0.74       |      3.25       |      8.67       |
|   SE7-32    |  yolov8_bmcv.soc  | yolov8s_int8_4b.bmodel  |      4.25       |      0.71       |      3.33       |      8.59       |
|   SE7-32    |  yolov8_bmcv.soc  |yolov8s_opt_fp32_1b.bmodel|      4.41       |      0.74       |      29.35      |      2.63       |
|   SE7-32    |  yolov8_bmcv.soc  |yolov8s_opt_fp16_1b.bmodel|      4.36       |      0.74       |      5.58       |      2.62       |
|   SE7-32    |  yolov8_bmcv.soc  |yolov8s_opt_int8_1b.bmodel|      4.39       |      0.74       |      2.90       |      2.62       |
|   SE7-32    |  yolov8_bmcv.soc  |yolov8s_opt_int8_4b.bmodel|      4.24       |      0.71       |      2.89       |      2.72       |
|   SE9-16    | yolov8_opencv.py  | yolov8s_fp32_1b.bmodel  |      23.35      |      29.68      |     169.78      |      6.93       |
|   SE9-16    | yolov8_opencv.py  | yolov8s_fp16_1b.bmodel  |      19.39      |      30.28      |      41.80      |      6.95       |
|   SE9-16    | yolov8_opencv.py  | yolov8s_int8_1b.bmodel  |      19.36      |      29.60      |      18.25      |      6.88       |
|   SE9-16    | yolov8_opencv.py  | yolov8s_int8_4b.bmodel  |      19.27      |      33.23      |      17.60      |      7.40       |
|   SE9-16    |  yolov8_bmcv.py   | yolov8s_fp32_1b.bmodel  |      4.45       |      4.99       |     165.95      |      6.98       |
|   SE9-16    |  yolov8_bmcv.py   | yolov8s_fp16_1b.bmodel  |      4.40       |      4.96       |      37.91      |      6.97       |
|   SE9-16    |  yolov8_bmcv.py   | yolov8s_int8_1b.bmodel  |      4.41       |      4.92       |      14.41      |      6.95       |
|   SE9-16    |  yolov8_bmcv.py   | yolov8s_int8_4b.bmodel  |      4.29       |      4.64       |      13.41      |      6.14       |
|   SE9-16    |  yolov8_bmcv.soc  | yolov8s_fp32_1b.bmodel  |      5.80       |      1.83       |     162.12      |      12.11      |
|   SE9-16    |  yolov8_bmcv.soc  | yolov8s_fp16_1b.bmodel  |      5.92       |      1.82       |      34.29      |      12.09      |
|   SE9-16    |  yolov8_bmcv.soc  | yolov8s_int8_1b.bmodel  |      5.85       |      1.83       |      10.82      |      12.10      |
|   SE9-16    |  yolov8_bmcv.soc  | yolov8s_int8_4b.bmodel  |      5.70       |      1.74       |      10.57      |      12.05      |
|   SE9-16    |  yolov8_bmcv.soc  |yolov8s_opt_fp32_1b.bmodel|      5.85       |      1.82       |     161.79      |      3.69       |
|   SE9-16    |  yolov8_bmcv.soc  |yolov8s_opt_fp16_1b.bmodel|      5.88       |      1.82       |      34.19      |      3.67       |
|   SE9-16    |  yolov8_bmcv.soc  |yolov8s_opt_int8_1b.bmodel|      5.87       |      1.82       |      8.10       |      3.67       |
|   SE9-16    |  yolov8_bmcv.soc  |yolov8s_opt_int8_4b.bmodel|      5.67       |      1.74       |      7.86       |      3.70       |
|   SE9-16    | yolov8_opencv.py  |yolov8s_fp32_1b_2core.bmodel|      35.37      |      29.56      |     100.48      |      6.94       |
|   SE9-16    | yolov8_opencv.py  |yolov8s_fp16_1b_2core.bmodel|      19.39      |      30.15      |      29.73      |      6.93       |
|   SE9-16    | yolov8_opencv.py  |yolov8s_int8_1b_2core.bmodel|      19.32      |      29.63      |      16.04      |      6.88       |
|   SE9-16    | yolov8_opencv.py  |yolov8s_int8_4b_2core.bmodel|      19.33      |      33.28      |      13.36      |      7.65       |
|   SE9-16    |  yolov8_bmcv.py   |yolov8s_fp32_1b_2core.bmodel|      4.49       |      4.98       |      96.83      |      6.96       |
|   SE9-16    |  yolov8_bmcv.py   |yolov8s_fp16_1b_2core.bmodel|      4.42       |      4.92       |      25.98      |      6.94       |
|   SE9-16    |  yolov8_bmcv.py   |yolov8s_int8_1b_2core.bmodel|      4.39       |      4.90       |      11.99      |      6.95       |
|   SE9-16    |  yolov8_bmcv.py   |yolov8s_int8_4b_2core.bmodel|      4.28       |      4.59       |      9.10       |      6.12       |
|   SE9-16    |  yolov8_bmcv.soc  |yolov8s_fp32_1b_2core.bmodel|      5.90       |      1.83       |      93.00      |      12.10      |
|   SE9-16    |  yolov8_bmcv.soc  |yolov8s_fp16_1b_2core.bmodel|      5.89       |      1.82       |      22.28      |      12.10      |
|   SE9-16    |  yolov8_bmcv.soc  |yolov8s_int8_1b_2core.bmodel|      5.92       |      1.83       |      8.35       |      12.18      |
|   SE9-16    |  yolov8_bmcv.soc  |yolov8s_int8_4b_2core.bmodel|      5.78       |      1.75       |      6.24       |      12.08      |
|   SE9-16    |  yolov8_bmcv.soc  |yolov8s_opt_fp32_1b_2core.bmodel|      5.92       |      1.84       |      92.15      |      3.72       |
|   SE9-16    |  yolov8_bmcv.soc  |yolov8s_opt_fp16_1b_2core.bmodel|      5.90       |      1.83       |      22.06      |      3.66       |
|   SE9-16    |  yolov8_bmcv.soc  |yolov8s_opt_int8_1b_2core.bmodel|      5.89       |      1.83       |      7.25       |      3.67       |
|   SE9-16    |  yolov8_bmcv.soc  |yolov8s_opt_int8_4b_2core.bmodel|      5.75       |      1.75       |      4.84       |      3.71       |
|    SE9-8    | yolov8_opencv.py  | yolov8s_fp32_1b.bmodel  |      25.22      |      30.41      |     169.19      |      7.01       |
|    SE9-8    | yolov8_opencv.py  | yolov8s_fp16_1b.bmodel  |      24.42      |      29.83      |      41.77      |      6.97       |
|    SE9-8    | yolov8_opencv.py  | yolov8s_int8_1b.bmodel  |      19.31      |      30.46      |      18.25      |      6.95       |
|    SE9-8    | yolov8_opencv.py  | yolov8s_int8_4b.bmodel  |      19.42      |      32.97      |      17.66      |      7.41       |
|    SE9-8    |  yolov8_bmcv.py   | yolov8s_fp32_1b.bmodel  |      4.34       |      4.78       |     165.23      |      7.05       |
|    SE9-8    |  yolov8_bmcv.py   | yolov8s_fp16_1b.bmodel  |      4.34       |      4.77       |      37.60      |      7.07       |
|    SE9-8    |  yolov8_bmcv.py   | yolov8s_int8_1b.bmodel  |      4.29       |      4.76       |      14.32      |      7.10       |
|    SE9-8    |  yolov8_bmcv.py   | yolov8s_int8_4b.bmodel  |      4.14       |      4.47       |      13.17      |      6.21       |
|    SE9-8    |  yolov8_bmcv.soc  | yolov8s_fp32_1b.bmodel  |      5.85       |      1.82       |     161.46      |      12.21      |
|    SE9-8    |  yolov8_bmcv.soc  | yolov8s_fp16_1b.bmodel  |      5.83       |      1.82       |      34.05      |      12.19      |
|    SE9-8    |  yolov8_bmcv.soc  | yolov8s_int8_1b.bmodel  |      5.77       |      1.82       |      10.71      |      12.20      |
|    SE9-8    |  yolov8_bmcv.soc  | yolov8s_int8_4b.bmodel  |      5.65       |      1.74       |      10.46      |      12.14      |
|    SE9-8    |  yolov8_bmcv.soc  |yolov8s_opt_fp32_1b.bmodel|      5.85       |      1.82       |     161.75      |      3.71       |
|    SE9-8    |  yolov8_bmcv.soc  |yolov8s_opt_fp16_1b.bmodel|      5.79       |      1.82       |      34.15      |      4.01       |
|    SE9-8    |  yolov8_bmcv.soc  |yolov8s_opt_int8_1b.bmodel|      5.70       |      1.82       |      7.77       |      3.68       |
|    SE9-8    |  yolov8_bmcv.soc  |yolov8s_opt_int8_4b.bmodel|      5.55       |      1.73       |      7.54       |      3.73       |
> **测试说明**：  
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. SE5-16/SE7-32的主控处理器均为8核CA53@2.3GHz，SE9-16为8核CA53@1.6GHz，SE9-8为6核CA53@1.6GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异，不同的阈值对后处理时间影响较大。 


## 8. FAQ
请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。