[简体中文](./README.md) | [English](./README_EN.md)

# YOLOv5

## 目录

- [YOLOv5](#yolov5)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
    - [2.1 目录结构说明](#21-目录结构说明)
    - [2.2 SDK特性](#22-sdk特性)
    - [2.3 算法特性](#23-算法特性)
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
  - [7. YOLOv5 cpu opt](#7-yolov5-cpu-opt)
    - [7.1 NMS优化项](#71-nms优化项)
    - [7.2 精度测试](#72-精度测试)
    - [7.3 性能测试](#73-性能测试)
  - [8. FAQ](#8-faq)
  
## 1. 简介
​YOLOv5是非常经典的基于anchor的One Stage目标检测算法，因其优秀的精度和速度表现，在工程实践应用中获得了非常广泛的应用。本例程对[​YOLOv5官方开源仓库](https://github.com/ultralytics/yolov5)v6.1版本的模型和算法进行移植，使之能在SOPHON BM1684/BM1684X/BM1688/CV186X上进行推理测试。

## 2. 特性

### 2.1 目录结构说明
```bash
├── cpp                   # 存放C++例程及其README
|   ├──README_EN.md     
|   ├──README.md      
|   ├──yolov5_bmcv        # 使用FFmpeg解码、BMCV前处理、BMRT推理的C++例程
|   └──yolov5_sail        # 使用SAIL解码、SAIL.BMCV前处理、SAIL推理的C++例程
├── docs                  # 存放本例程专用文档，如ONNX导出、移植常见问题等
├── pics                  # 存放README等说明文档中用到的图片
├── python                # 存放Python例程及其README
|   ├──README_EN.md 
|   ├──README.md 
|   ├──yolov5_bmcv.py     # 使用SAIL解码、SAIL.BMCV前处理、SAIL推理的Python例程
|   ├──yolov5_opencv.py   # 使用OpenCV解码、OpenCV前处理、SAIL推理的Python例程
|   └──...                # Python例程共用功能的封装。
├── README_EN.md          # 本例程的英文指南
├── README.md             # 本例程的中文指南
├── scripts               # 存放模型编译、数据下载、自动测试等shell脚本
└── tools                 # 存放精度测试、性能比对等python脚本
```

### 2.2 SDK特性
* 支持BM1688/CV186X(SoC)、BM1684X(x86 PCIe、SoC、riscv PCIe)、BM1684(x86 PCIe、SoC、arm PCIe)
* 支持FP32、FP16(BM1684X/BM1688/CV186X)、INT8模型编译和推理
* 支持基于BMCV预处理的C++推理
* 支持基于OpenCV和BMCV预处理的Python推理
* 支持单batch和多batch模型推理
* 支持1个输出和3个输出模型推理
* 支持图片和视频测试
* 支持NMS后处理算法软件加速

### 2.3 算法特性
考虑到YOLOv5的泛用性，针对算法的性能我们提出了许多优化的版本可供您选择：

python 版本
1. `sophon-demo/sample/YOLOv5/python`,例程源码位于本仓库，分别基于`bmcv`以及`opencv`的接口,实现的简易例程，用于测试模型的准确性，**不推荐用于性能评测**；
2. `sophon-demo/sample/YOLOv5_opt/python`,例程源码位于本仓库,**仅支持1684x**,将yolov5解码层以及nms操作使用TPU实现，提高端到端性能；
3. `sophon-sail/sample/python/yolov5_multi_3output_pic.py`,例程源码位于SDK中`sophon-sail`，使用python调用C++封装的接口,从而将解码，前处理，推理，后处理放在不同的线程上，提高整体的性能
4. 《TPU-MLIR快速入门手册》第8、9小节，使用TPU做前处理、后处理，需要您参考对应的[文档](https://doc.sophgo.com/sdk-docs/v23.07.01/docs_latest_release/docs/tpu-mlir/quick_start/html/08_fuse_preprocess.html),编写对应的例程，将算法的前处理，后处理全部放在模型里面，提高端到端的性能；
5. 使用python多进程库`multiprocessing`,将耗时较多的的接口，使用多进程的方式调用，可提高整体吞吐量；
6. `sophon-demo/sample/YOLOv5_fuse/python`，将前后处理融合进模型，**显著提升端到端性能，SE7&SE9强烈推荐**，对SDK版本要求较高，建议使用官网最新的libsophon驱动和mlir工具链。

c++版本
1. `sophon-demo/sample/YOLOv5/cpp`,例程源码位于本仓库内，分别基于`bmcv`,`sail`的接口实现的简易例程，用于准确性的验证
2. `sophon-demo/sample/YOLOv5_opt/cpp`,例程源码位于本仓库,**仅支持1684x**,将yolov5解码层以及nms操作使用TPU实现，提高端到端性能；
3. `sophon-stream/samples/yolov5`,例程源码位于SDK（V23.10.01及以上版本）中`sophon-stream`,将前处理、推理、后处理放在不同的线程上，大幅提高整体性能；
4. `sophon-pipeline/examples/yolov5`,[例程源码链接](https://github.com/sophgo/sophon-pipeline),基于线程池实现整个算法推理过程，提高整体性能；
5. 《TPU-MLIR快速入门手册》第8、9小节，使用TPU做前处理、后处理，提高端到端的性能
6. `sophon-demo/sample/YOLOv5_fuse/cpp`，将前后处理融合进模型，**显著提升端到端性能，SE7&SE9强烈推荐**，对SDK版本要求较高，建议使用官网最新的libsophon驱动和mlir工具链。

> **注意：**  
> 本例程支持三输出以及单输出模型，其中单输出模型性能更高，但是量化需要设置敏感层；三输出模型量化简单，在**用于验证模型准确性时，推荐使用三输出模型**

## 3. 数据准备与模型编译

### 3.1 数据准备

​本例程在`scripts`目录下提供了相关模型和数据的下载脚本`download.sh`，**如果您希望自己准备模型和数据集，可以跳过本小节，参考[3.2 模型编译](#32-模型编译)进行模型转换。**

```bash
chmod -R +x scripts/
./scripts/download.sh
```

下载的模型包括：
```
./models
├── BM1684
│   ├── yolov5s_v6.1_3output_fp32_1b.bmodel       # 使用TPU-MLIR编译，用于BM1684的FP32 BModel，batch_size=1
│   ├── yolov5s_v6.1_3output_int8_1b.bmodel       # 使用TPU-MLIR编译，用于BM1684的INT8 BModel，batch_size=1
│   └── yolov5s_v6.1_3output_int8_4b.bmodel       # 使用TPU-MLIR编译，用于BM1684的INT8 BModel，batch_size=4
├── BM1684X    
│   ├── yolov5s_v6.1_3output_fp32_1b.bmodel       # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=1
│   ├── yolov5s_v6.1_3output_fp16_1b.bmodel       # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=1
│   ├── yolov5s_v6.1_3output_int8_1b.bmodel       # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=1
│   └── yolov5s_v6.1_3output_int8_4b.bmodel       # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=4
├── BM1688
│   ├── yolov5s_v6.1_3output_fp16_1b.bmodel       # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1, num_core=1
│   ├── yolov5s_v6.1_3output_fp32_1b.bmodel       # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1, num_core=1
│   ├── yolov5s_v6.1_3output_int8_1b.bmodel       # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=1, num_core=1
│   ├── yolov5s_v6.1_3output_int8_4b.bmodel       # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=1, num_core=1
│   ├── yolov5s_v6.1_3output_fp16_1b_2core.bmodel # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1, num_core=2
│   ├── yolov5s_v6.1_3output_fp32_1b_2core.bmodel # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1, num_core=2
│   ├── yolov5s_v6.1_3output_int8_1b_2core.bmodel # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=1, num_core=2
│   └── yolov5s_v6.1_3output_int8_4b_2core.bmodel # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=4, num_core=2
├── CV186X
│   ├── yolov5s_v6.1_3output_fp16_1b.bmodel       # 使用TPU-MLIR编译，用于CV186X的FP16 BModel，batch_size=1
│   ├── yolov5s_v6.1_3output_fp32_1b.bmodel       # 使用TPU-MLIR编译，用于CV186X的FP32 BModel，batch_size=1
│   ├── yolov5s_v6.1_3output_int8_1b.bmodel       # 使用TPU-MLIR编译，用于CV186X的INT8 BModel，batch_size=1
│   └── yolov5s_v6.1_3output_int8_4b.bmodel       # 使用TPU-MLIR编译，用于CV186X的INT8 BModel，batch_size=1
├── BM1684_ext                                # 相关单输出模型，此处没有benchmark，用户自行使用。
├── BM1684X_ext                               # 相关单输出模型，此处没有benchmark，用户自行使用。
│── torch
│   ├── yolov5m_v6.1_1output_torchscript.pt   # 相关单输出模型，此处没有benchmark，用户自行使用。
│   └── yolov5s_v6.1_3output.torchscript.pt   # trace后的torchscript模型
└── onnx
    ├── yolov5m_v6.1_1output_1b.onnx          # 相关单输出模型，此处没有benchmark，用户自行使用。
    ├── yolov5m_v6.1_1output_4b.onnx          # 相关单输出模型，此处没有benchmark，用户自行使用。
    └── yolov5s_v6.1_3output.onnx             # 导出的onnx动态模型       
```
下载的数据包括：
```
./datasets
├── test                                      # 测试图片
├── test_car_person_1080P.mp4                 # 测试视频
├── coco.names                                # coco类别名文件
├── coco128                                   # coco128数据集，用于模型量化
└── coco                                      
    ├── val2017_1000                          # coco val2017_1000数据集：coco val2017中随机抽取的1000张样本
    └── instances_val2017_1000.json           # coco val2017_1000数据集标签文件，用于计算精度评价指标  
```

### 3.2 模型编译

**如果您不编译模型，只想直接使用下载的数据集和模型，可以跳过本小节。**

源模型需要编译成BModel才能在SOPHON TPU上运行，源模型在编译前要导出成onnx模型，如果您使用的TPU-MLIR版本>=v1.3.0（即官网v23.07.01），也可以直接使用torchscript模型。具体可参考[YOLOv5模型导出](./docs/YOLOv5_Export_Guide.md)。​同时，您需要准备用于测试的数据集，如果量化模型，还要准备用于量化的数据集。

建议使用TPU-MLIR编译BModel，模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录，并使用本例程提供的脚本将onnx模型编译为BModel。脚本中命令的详细说明可参考《TPU-MLIR开发手册》(请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的SDK中获取)。

- 生成FP32 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684/BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684 #bm1684x/bm1688/cv186x
```

​执行上述命令会在`models/BM1684`等文件夹下生成`yolov5s_v6.1_3output_fp32_1b.bmodel`文件，即转换好的FP32 BModel。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x #bm1688/cv186x
```

​执行上述命令会在`models/BM1684X/`等文件夹下生成`yolov5s_v6.1_3output_fp16_1b.bmodel`文件，即转换好的FP16 BModel。

- 生成INT8 BModel

​本例程在`scripts`目录下提供了量化INT8 BModel的脚本，请注意修改`gen_int8bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台（**支持BM1684/BM1684X/BM1688/CV186X**），如：

```shell
./scripts/gen_int8bmodel_mlir.sh bm1684 #bm1684x/bm1688/cv186x
```

​上述脚本会在`models/BM1684`等文件夹下生成`yolov5s_v6.1_3output_int8_1b.bmodel`等文件，即转换好的INT8 BModel。


## 4. 例程测试
- [C++例程](./cpp/README.md)
- [Python例程](./python/README.md)

## 5. 精度测试
### 5.1 测试方法

首先，参考[C++例程](cpp/README.md#32-测试图片)或[Python例程](python/README.md#22-测试图片)推理要测试的数据集，生成预测的json文件，注意修改数据集(datasets/coco/val2017_1000)和相关参数(conf_thresh=0.001、nms_thresh=0.6)。  
然后，使用`tools`目录下的`eval_coco.py`脚本，将测试生成的json文件与测试集标签json文件进行对比，计算出目标检测的评价指标，命令如下：
```bash
# 安装pycocotools，若已安装请跳过
pip3 install pycocotools
# 请根据实际情况修改程序路径和json文件路径
python3 tools/eval_coco.py --gt_path datasets/coco/instances_val2017_1000.json --result_json results/yolov5s_v6.1_3output_fp32_1b.bmodel_val2017_1000_opencv_python_result.json
```
### 5.2 测试结果
CPP设置`--use_cpu_opt=false`或python不设置`--use_cpu_opt`进行测试，在`datasets/coco/val2017_1000`数据集上，精度测试结果如下：
|   测试平台    |      测试程序     |              测试模型               |AP@IoU=0.5:0.95|AP@IoU=0.5|
| ------------ | ---------------- | ----------------------------------- | ------------- | -------- |
| SE5-16       | yolov5_opencv.py   | yolov5s_v6.1_3output_fp32_1b.bmodel      |    0.377 |    0.580 |
| SE5-16       | yolov5_opencv.py   | yolov5s_v6.1_3output_int8_1b.bmodel      |    0.344 |    0.553 |
| SE5-16       | yolov5_opencv.py   | yolov5s_v6.1_3output_int8_4b.bmodel      |    0.344 |    0.553 |
| SE5-16       | yolov5_bmcv.py     | yolov5s_v6.1_3output_fp32_1b.bmodel      |    0.373 |    0.573 |
| SE5-16       | yolov5_bmcv.py     | yolov5s_v6.1_3output_int8_1b.bmodel      |    0.337 |    0.544 |
| SE5-16       | yolov5_bmcv.py     | yolov5s_v6.1_3output_int8_4b.bmodel      |    0.337 |    0.544 |
| SE5-16       | yolov5_bmcv.soc    | yolov5s_v6.1_3output_fp32_1b.bmodel      |    0.375 |    0.572 |
| SE5-16       | yolov5_bmcv.soc    | yolov5s_v6.1_3output_int8_1b.bmodel      |    0.338 |    0.544 |
| SE5-16       | yolov5_bmcv.soc    | yolov5s_v6.1_3output_int8_4b.bmodel      |    0.338 |    0.544 |
| SE5-16       | yolov5_sail.soc    | yolov5s_v6.1_3output_fp32_1b.bmodel      |    0.375 |    0.572 |
| SE5-16       | yolov5_sail.soc    | yolov5s_v6.1_3output_int8_1b.bmodel      |    0.338 |    0.544 |
| SE5-16       | yolov5_sail.soc    | yolov5s_v6.1_3output_int8_4b.bmodel      |    0.338 |    0.544 |
| SE7-32       | yolov5_opencv.py   | yolov5s_v6.1_3output_fp32_1b.bmodel      |    0.377 |    0.580 |
| SE7-32       | yolov5_opencv.py   | yolov5s_v6.1_3output_fp16_1b.bmodel      |    0.377 |    0.580 |
| SE7-32       | yolov5_opencv.py   | yolov5s_v6.1_3output_int8_1b.bmodel      |    0.361 |    0.570 |
| SE7-32       | yolov5_opencv.py   | yolov5s_v6.1_3output_int8_4b.bmodel      |    0.361 |    0.570 |
| SE7-32       | yolov5_bmcv.py     | yolov5s_v6.1_3output_fp32_1b.bmodel      |    0.373 |    0.573 |
| SE7-32       | yolov5_bmcv.py     | yolov5s_v6.1_3output_fp16_1b.bmodel      |    0.373 |    0.573 |
| SE7-32       | yolov5_bmcv.py     | yolov5s_v6.1_3output_int8_1b.bmodel      |    0.356 |    0.563 |
| SE7-32       | yolov5_bmcv.py     | yolov5s_v6.1_3output_int8_4b.bmodel      |    0.356 |    0.563 |
| SE7-32       | yolov5_bmcv.soc    | yolov5s_v6.1_3output_fp32_1b.bmodel      |    0.374 |    0.572 |
| SE7-32       | yolov5_bmcv.soc    | yolov5s_v6.1_3output_fp16_1b.bmodel      |    0.374 |    0.572 |
| SE7-32       | yolov5_bmcv.soc    | yolov5s_v6.1_3output_int8_1b.bmodel      |    0.357 |    0.562 |
| SE7-32       | yolov5_bmcv.soc    | yolov5s_v6.1_3output_int8_4b.bmodel      |    0.357 |    0.562 |
| SE7-32       | yolov5_sail.soc    | yolov5s_v6.1_3output_fp32_1b.bmodel      |    0.374 |    0.572 |
| SE7-32       | yolov5_sail.soc    | yolov5s_v6.1_3output_fp16_1b.bmodel      |    0.374 |    0.572 |
| SE7-32       | yolov5_sail.soc    | yolov5s_v6.1_3output_int8_1b.bmodel      |    0.357 |    0.562 |
| SE7-32       | yolov5_sail.soc    | yolov5s_v6.1_3output_int8_4b.bmodel      |    0.357 |    0.562 |
| SE9-16       | yolov5_opencv.py   | yolov5s_v6.1_3output_fp32_1b.bmodel      |    0.377 |    0.580 |
| SE9-16       | yolov5_opencv.py   | yolov5s_v6.1_3output_fp16_1b.bmodel      |    0.377 |    0.580 |
| SE9-16       | yolov5_opencv.py   | yolov5s_v6.1_3output_int8_1b.bmodel      |    0.358 |    0.567 |
| SE9-16       | yolov5_opencv.py   | yolov5s_v6.1_3output_int8_4b.bmodel      |    0.358 |    0.567 |
| SE9-16       | yolov5_bmcv.py     | yolov5s_v6.1_3output_fp32_1b.bmodel      |    0.373 |    0.573 |
| SE9-16       | yolov5_bmcv.py     | yolov5s_v6.1_3output_fp16_1b.bmodel      |    0.373 |    0.573 |
| SE9-16       | yolov5_bmcv.py     | yolov5s_v6.1_3output_int8_1b.bmodel      |    0.355 |    0.565 |
| SE9-16       | yolov5_bmcv.py     | yolov5s_v6.1_3output_int8_4b.bmodel      |    0.355 |    0.565 |
| SE9-16       | yolov5_bmcv.soc    | yolov5s_v6.1_3output_fp32_1b.bmodel      |    0.374 |    0.573 |
| SE9-16       | yolov5_bmcv.soc    | yolov5s_v6.1_3output_fp16_1b.bmodel      |    0.374 |    0.572 |
| SE9-16       | yolov5_bmcv.soc    | yolov5s_v6.1_3output_int8_1b.bmodel      |    0.354 |    0.565 |
| SE9-16       | yolov5_bmcv.soc    | yolov5s_v6.1_3output_int8_4b.bmodel      |    0.354 |    0.565 |
| SE9-16       | yolov5_sail.soc    | yolov5s_v6.1_3output_fp32_1b.bmodel      |    0.374 |    0.573 |
| SE9-16       | yolov5_sail.soc    | yolov5s_v6.1_3output_fp16_1b.bmodel      |    0.374 |    0.572 |
| SE9-16       | yolov5_sail.soc    | yolov5s_v6.1_3output_int8_1b.bmodel      |    0.354 |    0.565 |
| SE9-16       | yolov5_sail.soc    | yolov5s_v6.1_3output_int8_4b.bmodel      |    0.354 |    0.565 |
| SE9-16       | yolov5_opencv.py   | yolov5s_v6.1_3output_fp32_1b_2core.bmodel |    0.377 |    0.580 |
| SE9-16       | yolov5_opencv.py   | yolov5s_v6.1_3output_fp16_1b_2core.bmodel |    0.377 |    0.580 |
| SE9-16       | yolov5_opencv.py   | yolov5s_v6.1_3output_int8_1b_2core.bmodel |    0.358 |    0.567 |
| SE9-16       | yolov5_opencv.py   | yolov5s_v6.1_3output_int8_4b_2core.bmodel |    0.358 |    0.567 |
| SE9-16       | yolov5_bmcv.py     | yolov5s_v6.1_3output_fp32_1b_2core.bmodel |    0.373 |    0.573 |
| SE9-16       | yolov5_bmcv.py     | yolov5s_v6.1_3output_fp16_1b_2core.bmodel |    0.373 |    0.573 |
| SE9-16       | yolov5_bmcv.py     | yolov5s_v6.1_3output_int8_1b_2core.bmodel |    0.355 |    0.565 |
| SE9-16       | yolov5_bmcv.py     | yolov5s_v6.1_3output_int8_4b_2core.bmodel |    0.355 |    0.565 |
| SE9-16       | yolov5_bmcv.soc    | yolov5s_v6.1_3output_fp32_1b_2core.bmodel |    0.374 |    0.573 |
| SE9-16       | yolov5_bmcv.soc    | yolov5s_v6.1_3output_fp16_1b_2core.bmodel |    0.374 |    0.572 |
| SE9-16       | yolov5_bmcv.soc    | yolov5s_v6.1_3output_int8_1b_2core.bmodel |    0.354 |    0.565 |
| SE9-16       | yolov5_bmcv.soc    | yolov5s_v6.1_3output_int8_4b_2core.bmodel |    0.354 |    0.565 |
| SE9-16       | yolov5_sail.soc    | yolov5s_v6.1_3output_fp32_1b_2core.bmodel |    0.374 |    0.573 |
| SE9-16       | yolov5_sail.soc    | yolov5s_v6.1_3output_fp16_1b_2core.bmodel |    0.374 |    0.572 |
| SE9-16       | yolov5_sail.soc    | yolov5s_v6.1_3output_int8_1b_2core.bmodel |    0.354 |    0.565 |
| SE9-16       | yolov5_sail.soc    | yolov5s_v6.1_3output_int8_4b_2core.bmodel |    0.354 |    0.565 |
| SE9-8        | yolov5_opencv.py   | yolov5s_v6.1_3output_fp32_1b.bmodel      |    0.377 |    0.580 |
| SE9-8        | yolov5_opencv.py   | yolov5s_v6.1_3output_fp16_1b.bmodel      |    0.377 |    0.580 |
| SE9-8        | yolov5_opencv.py   | yolov5s_v6.1_3output_int8_1b.bmodel      |    0.358 |    0.567 |
| SE9-8        | yolov5_opencv.py   | yolov5s_v6.1_3output_int8_4b.bmodel      |    0.358 |    0.567 |
| SE9-8        | yolov5_bmcv.py     | yolov5s_v6.1_3output_fp32_1b.bmodel      |    0.373 |    0.573 |
| SE9-8        | yolov5_bmcv.py     | yolov5s_v6.1_3output_fp16_1b.bmodel      |    0.373 |    0.573 |
| SE9-8        | yolov5_bmcv.py     | yolov5s_v6.1_3output_int8_1b.bmodel      |    0.355 |    0.565 |
| SE9-8        | yolov5_bmcv.py     | yolov5s_v6.1_3output_int8_4b.bmodel      |    0.355 |    0.565 |
| SE9-8        | yolov5_bmcv.soc    | yolov5s_v6.1_3output_fp32_1b.bmodel      |    0.374 |    0.572 |
| SE9-8        | yolov5_bmcv.soc    | yolov5s_v6.1_3output_fp16_1b.bmodel      |    0.374 |    0.572 |
| SE9-8        | yolov5_bmcv.soc    | yolov5s_v6.1_3output_int8_1b.bmodel      |    0.354 |    0.564 |
| SE9-8        | yolov5_bmcv.soc    | yolov5s_v6.1_3output_int8_4b.bmodel      |    0.354 |    0.564 |
| SE9-8        | yolov5_sail.soc    | yolov5s_v6.1_3output_fp32_1b.bmodel      |    0.374 |    0.572 |
| SE9-8        | yolov5_sail.soc    | yolov5s_v6.1_3output_fp16_1b.bmodel      |    0.374 |    0.572 |
| SE9-8        | yolov5_sail.soc    | yolov5s_v6.1_3output_int8_1b.bmodel      |    0.354 |    0.564 |
| SE9-8        | yolov5_sail.soc    | yolov5s_v6.1_3output_int8_4b.bmodel      |    0.354 |    0.564 |

> **测试说明**：  
> 1. 由于sdk版本之间可能存在差异，实际运行结果与本表有<0.01的精度误差是正常的；
> 2. AP@IoU=0.5:0.95为area=all对应的指标；
> 3. 在搭载了相同TPU和SOPHONSDK的PCIe或SoC平台上，相同程序的精度一致，SE5系列对应BM1684，SE7系列对应BM1684X，SE9系列中，SE9-16对应BM1688，SE9-8对应CV186X；

## 6. 性能测试
### 6.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684/yolov5s_v6.1_3output_fp32_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

|                  测试模型                         | calculate time(ms) |
| -------------------------------------------       | ----------------- |
| BM1684/yolov5s_v6.1_3output_fp32_1b.bmodel |          22.41  |
| BM1684/yolov5s_v6.1_3output_int8_1b.bmodel |          11.26  |
| BM1684/yolov5s_v6.1_3output_int8_4b.bmodel |           6.04  |
| BM1684X/yolov5s_v6.1_3output_fp32_1b.bmodel|          21.66  |
| BM1684X/yolov5s_v6.1_3output_fp16_1b.bmodel|           7.37  |
| BM1684X/yolov5s_v6.1_3output_int8_1b.bmodel|           3.51  |
| BM1684X/yolov5s_v6.1_3output_int8_4b.bmodel|           3.34  |
| BM1688/yolov5s_v6.1_3output_fp32_1b.bmodel|         101.57  |
| BM1688/yolov5s_v6.1_3output_fp16_1b.bmodel|          29.92  |
| BM1688/yolov5s_v6.1_3output_int8_1b.bmodel|           9.33  |
| BM1688/yolov5s_v6.1_3output_int8_4b.bmodel|           8.90  |
| BM1688/yolov5s_v6.1_3output_fp32_1b_2core.bmodel|          66.89  |
| BM1688/yolov5s_v6.1_3output_fp16_1b_2core.bmodel|          20.62  |
| BM1688/yolov5s_v6.1_3output_int8_1b_2core.bmodel|           8.53  |
| BM1688/yolov5s_v6.1_3output_int8_4b_2core.bmodel|           6.87  |
| CV186X/yolov5s_v6.1_3output_fp32_1b.bmodel|         100.68  |
| CV186X/yolov5s_v6.1_3output_fp16_1b.bmodel|          29.93  |
| CV186X/yolov5s_v6.1_3output_int8_1b.bmodel|           8.18  |
| CV186X/yolov5s_v6.1_3output_int8_4b.bmodel|           7.90  |

> **测试说明**：  
> 1. 性能测试结果具有一定的波动性；
> 2. `calculate time`已折算为平均每张图片的推理时间；
> 3. SoC和PCIe的测试结果基本一致。

### 6.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++和Python例程打印的时间已经折算为单张图片的处理时间。

CPP设置`--use_cpu_opt=false`或python不设置`--use_cpu_opt`进行测试，在不同的测试平台上，使用不同的例程、模型测试`datasets/coco/val2017_1000`，conf_thresh=0.5，nms_thresh=0.5，性能测试结果如下：
|    测试平台  |     测试程序      |             测试模型                |decode_time    |preprocess_time  |inference_time   |postprocess_time| 
| ----------- | ---------------- | ----------------------------------- | --------      | ---------       | ---------        | --------- |
|   SE5-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_fp32_1b.bmodel|      15.08      |      21.95      |      31.40      |     107.61      |
|   SE5-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_int8_1b.bmodel|      15.07      |      26.16      |      34.45      |     110.48      |
|   SE5-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_int8_4b.bmodel|      15.03      |      23.94      |      27.53      |     111.78      |
|   SE5-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_fp32_1b.bmodel|      3.61       |      2.83       |      29.06      |     106.85      |
|   SE5-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_int8_1b.bmodel|      3.59       |      2.31       |      17.92      |     106.40      |
|   SE5-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_int8_4b.bmodel|      3.44       |      2.13       |      11.82      |     110.71      |
|   SE5-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_fp32_1b.bmodel|      4.87       |      1.54       |      22.33      |      15.68      |
|   SE5-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_int8_1b.bmodel|      4.85       |      1.53       |      11.20      |      15.66      |
|   SE5-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_int8_4b.bmodel|      4.75       |      1.47       |      6.03       |      15.64      |
|   SE5-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_fp32_1b.bmodel|      3.23       |      3.04       |      23.31      |      14.07      |
|   SE5-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_int8_1b.bmodel|      3.22       |      1.80       |      12.21      |      13.93      |
|   SE5-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_int8_4b.bmodel|      3.08       |      1.71       |      6.88       |      13.80      |
|   SE7-32    | yolov5_opencv.py  |yolov5s_v6.1_3output_fp32_1b.bmodel|      15.09      |      27.82      |      33.27      |     108.98      |
|   SE7-32    | yolov5_opencv.py  |yolov5s_v6.1_3output_fp16_1b.bmodel|      15.01      |      27.27      |      19.10      |     109.18      |
|   SE7-32    | yolov5_opencv.py  |yolov5s_v6.1_3output_int8_1b.bmodel|      15.08      |      27.02      |      15.18      |     109.33      |
|   SE7-32    | yolov5_opencv.py  |yolov5s_v6.1_3output_int8_4b.bmodel|      14.99      |      25.01      |      13.31      |     108.20      |
|   SE7-32    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_fp32_1b.bmodel|      3.09       |      2.35       |      28.98      |     103.87      |
|   SE7-32    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_fp16_1b.bmodel|      3.09       |      2.34       |      14.75      |     103.75      |
|   SE7-32    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_int8_1b.bmodel|      3.08       |      2.34       |      10.92      |     103.89      |
|   SE7-32    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_int8_4b.bmodel|      2.93       |      2.16       |      9.82       |     108.36      |
|   SE7-32    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_fp32_1b.bmodel|      4.32       |      0.74       |      21.63      |      15.91      |
|   SE7-32    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_fp16_1b.bmodel|      4.32       |      0.74       |      7.38       |      15.94      |
|   SE7-32    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_int8_1b.bmodel|      4.33       |      0.74       |      3.48       |      15.94      |
|   SE7-32    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_int8_4b.bmodel|      4.17       |      0.71       |      3.32       |      15.73      |
|   SE7-32    |  yolov5_sail.soc  |yolov5s_v6.1_3output_fp32_1b.bmodel|      2.71       |      2.58       |      22.61      |      14.15      |
|   SE7-32    |  yolov5_sail.soc  |yolov5s_v6.1_3output_fp16_1b.bmodel|      2.71       |      2.59       |      8.35       |      14.19      |
|   SE7-32    |  yolov5_sail.soc  |yolov5s_v6.1_3output_int8_1b.bmodel|      2.70       |      2.59       |      4.45       |      14.18      |
|   SE7-32    |  yolov5_sail.soc  |yolov5s_v6.1_3output_int8_4b.bmodel|      2.56       |      2.50       |      4.20       |      14.06      |
|   SE9-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_fp32_1b.bmodel|      14.10      |      36.40      |     112.48      |     151.18      |
|   SE9-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_fp16_1b.bmodel|      9.82       |      35.82      |      41.96      |     150.27      |
|   SE9-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_int8_1b.bmodel|      9.60       |      36.77      |      21.98      |     150.78      |
|   SE9-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_int8_4b.bmodel|      9.44       |      33.33      |      19.38      |     152.33      |
|   SE9-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_fp32_1b.bmodel|      4.53       |      4.85       |     107.05      |     143.24      |
|   SE9-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_fp16_1b.bmodel|      4.53       |      4.86       |      36.74      |     143.46      |
|   SE9-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_int8_1b.bmodel|      4.53       |      4.85       |      16.87      |     143.56      |
|   SE9-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_int8_4b.bmodel|      4.40       |      4.54       |      14.93      |     149.27      |
|   SE9-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_fp32_1b.bmodel|      5.95       |      1.79       |      97.61      |      22.25      |
|   SE9-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_fp16_1b.bmodel|      5.97       |      1.79       |      27.37      |      22.22      |
|   SE9-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_int8_1b.bmodel|      5.96       |      1.79       |      7.14       |      22.25      |
|   SE9-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_int8_4b.bmodel|      5.81       |      1.71       |      7.03       |      21.98      |
|   SE9-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_fp32_1b.bmodel|      3.97       |      5.02       |     100.10      |      19.79      |
|   SE9-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_fp16_1b.bmodel|      3.97       |      5.00       |      29.84      |      19.77      |
|   SE9-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_int8_1b.bmodel|      3.94       |      5.00       |      9.60       |      19.76      |
|   SE9-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_int8_4b.bmodel|      3.79       |      4.76       |      9.29       |      19.63      |
|   SE9-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_fp32_1b_2core.bmodel|      9.51       |      36.35      |      67.50      |     150.78      |
|   SE9-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_fp16_1b_2core.bmodel|      9.41       |      35.45      |      32.00      |     150.56      |
|   SE9-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_int8_1b_2core.bmodel|      9.53       |      36.07      |      20.61      |     150.55      |
|   SE9-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_int8_4b_2core.bmodel|      9.43       |      32.73      |      17.31      |     152.09      |
|   SE9-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_fp32_1b_2core.bmodel|      4.54       |      4.86       |      62.00      |     143.29      |
|   SE9-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_fp16_1b_2core.bmodel|      4.54       |      4.86       |      27.86      |     143.22      |
|   SE9-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_int8_1b_2core.bmodel|      4.51       |      4.87       |      15.66      |     143.12      |
|   SE9-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_int8_4b_2core.bmodel|      4.39       |      4.52       |      12.94      |     149.71      |
|   SE9-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_fp32_1b_2core.bmodel|      5.96       |      1.79       |      52.70      |      22.23      |
|   SE9-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_fp16_1b_2core.bmodel|      5.99       |      1.79       |      18.09      |      22.24      |
|   SE9-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_int8_1b_2core.bmodel|      5.96       |      1.79       |      6.33       |      22.24      |
|   SE9-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_int8_4b_2core.bmodel|      5.79       |      1.71       |      5.00       |      21.99      |
|   SE9-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_fp32_1b_2core.bmodel|      3.98       |      5.01       |      55.17      |      19.79      |
|   SE9-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_fp16_1b_2core.bmodel|      3.98       |      5.01       |      20.55      |      19.80      |
|   SE9-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_int8_1b_2core.bmodel|      3.96       |      5.01       |      8.80       |      19.83      |
|   SE9-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_int8_4b_2core.bmodel|      3.78       |      4.75       |      7.26       |      19.64      |
|    SE9-8    | yolov5_opencv.py  |yolov5s_v6.1_3output_fp32_1b.bmodel|      20.99      |      36.78      |     112.86      |     151.88      |
|    SE9-8    | yolov5_opencv.py  |yolov5s_v6.1_3output_fp16_1b.bmodel|      20.48      |      36.45      |      42.31      |     151.69      |
|    SE9-8    | yolov5_opencv.py  |yolov5s_v6.1_3output_int8_1b.bmodel|      19.23      |      34.47      |      20.47      |     149.72      |
|    SE9-8    | yolov5_opencv.py  |yolov5s_v6.1_3output_int8_4b.bmodel|      19.31      |      33.41      |      18.75      |     154.23      |
|    SE9-8    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_fp32_1b.bmodel|      4.11       |      4.73       |     107.17      |     144.27      |
|    SE9-8    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_fp16_1b.bmodel|      4.05       |      4.71       |      36.70      |     144.19      |
|    SE9-8    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_int8_1b.bmodel|      4.12       |      4.76       |      15.80      |     144.34      |
|    SE9-8    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_int8_4b.bmodel|      3.94       |      4.42       |      14.18      |     150.38      |
|    SE9-8    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_fp32_1b.bmodel|      5.56       |      1.80       |      97.75      |      22.42      |
|    SE9-8    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_fp16_1b.bmodel|      5.56       |      1.79       |      27.33      |      22.41      |
|    SE9-8    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_int8_1b.bmodel|      5.54       |      1.79       |      6.35       |      22.45      |
|    SE9-8    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_int8_4b.bmodel|      5.39       |      1.72       |      6.22       |      22.14      |
|    SE9-8    |  yolov5_sail.soc  |yolov5s_v6.1_3output_fp32_1b.bmodel|      3.58       |      4.83       |     100.27      |      20.00      |
|    SE9-8    |  yolov5_sail.soc  |yolov5s_v6.1_3output_fp16_1b.bmodel|      3.60       |      4.82       |      29.83      |      19.99      |
|    SE9-8    |  yolov5_sail.soc  |yolov5s_v6.1_3output_int8_1b.bmodel|      3.62       |      4.82       |      8.84       |      20.04      |
|    SE9-8    |  yolov5_sail.soc  |yolov5s_v6.1_3output_int8_4b.bmodel|      3.45       |      4.63       |      8.53       |      19.78      |


> **测试说明**：  
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. SE5-16/SE7-32的主控处理器均为8核CA53@2.3GHz，SE9-16为8核CA53@1.6GHz，SE9-8为6核CA53@1.6GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异，不同的阈值对后处理时间影响较大。 

## 7. YOLOv5 cpu opt
本部分基于上述YOLOv5，优化了YOLOv5后处理NMS算法。下面主要说明NMS后处理算法优化的内容和优化后性能精度结果。

### 7.1 NMS优化项
* 提前噪声anchor的过滤，放在其他所有操作前，后续操作只需要处理数量显著减少的候选框
* 通过设置新阈值来优化掉anchor过滤中大量的sigmoid计算
* 优化存储减少数据遍历，在解码输出时仅仅保留候选框坐标、置信度、最高类别分数和对应索引
* 增大conf_thresh的值，过滤更多的噪声框
* 去除其他一些冗余计算

优化后NMS算法的时间瓶颈点在于模型输出的map大小，若尝试降低输出的map的高宽或通道数能够进一步降低NMS时间。
 
### 7.2 精度测试
在SE5-16上，使用不同的例程、模型测试`datasets/coco/val2017_1000`，阈值使用`conf_thresh=0.001，nms_thresh=0.6`，cpp设置`--use_cpu_opt=true`或python设置`--use_cpu_opt`，精度测试结果如下：
|   测试平台    |      测试程序     |              测试模型               |AP@IoU=0.5:0.95|AP@IoU=0.5|
| ------------ | ---------------- | ----------------------------------- | ------------- | -------- |
| SE5-16       | yolov5_opencv.py | yolov5s_v6.1_3output_fp32_1b.bmodel |    0.373      |    0.579 |
| SE5-16       | yolov5_bmcv.py   | yolov5s_v6.1_3output_fp32_1b.bmodel |    0.370      |    0.572 |
| SE5-16       | yolov5_bmcv.soc  | yolov5s_v6.1_3output_fp32_1b.bmodel |    0.375      |    0.573 |
| SE5-16       | yolov5_sail.soc  | yolov5s_v6.1_3output_fp32_1b.bmodel |    0.375      |    0.573 |

> **测试说明**：  
> 1. 此处适用6.2章节的测试说明；
> 2. 后处理加速不涉及硬件加速，此处只提供SE5-16平台、fp32模型的测试数据；

### 7.3 性能测试
在SE5-16上，使用不同的例程、模型测试`datasets/coco/val2017_1000`，阈值使用`conf_thresh=0.5，nms_thresh=0.5`，cpp设置`--use_cpu_opt=true`或python设置`--use_cpu_opt`，精度测试结果如下：
|    测试平台  |     测试程序      |             测试模型                |decode_time    |preprocess_time  |inference_time   |postprocess_time| 
| ----------- | ---------------- | ----------------------------------- | --------      | ---------       | ---------        | ---------      |
|   SE5-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_fp32_1b.bmodel|      14.99      |      21.52      |      43.84      |      16.83      |
|   SE5-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_fp32_1b.bmodel|      3.60       |      2.85       |      24.29      |      16.87      |
|   SE5-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_fp32_1b.bmodel|      4.88       |      1.54       |      22.33      |      6.17       |
|   SE5-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_fp32_1b.bmodel|      3.23       |      3.03       |      23.31      |      4.49       |

> **测试说明**：  
> 1. 此处适用7.2章节的测试说明；
> 2. 后处理加速不涉及硬件加速，此处只提供SE5-16平台、fp32模型的测试数据；
> 3. 可以通过提高`conf_thresh`参数值，或者使用单类NMS（即cpp例程设置`yolov5.cpp`文件中的宏`USE_MULTICLASS_NMS 0`或python例程设置文件`yolov5_opencv.py`、`yolov5_bmcv.py`中的YOLOv5类成员变量`self.multi_label=False`）来进一步提升后处理性能。

## 8. FAQ
YOLOv5移植相关问题可参考[YOLOv5常见问题](./docs/YOLOv5_Common_Problems.md)，其他问题请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。