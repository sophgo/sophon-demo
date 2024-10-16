# YOLOv11

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
​YOLOv11是YOLO系列的的一个重大更新版本，它抛弃了以往的YOLO系类模型使用的Anchor-Base，采用了Anchor-Free的思想。YOLOv11建立在YOLO系列成功的基础上，通过对网络结构的改造，进一步提升其性能和灵活性。本例程对[​YOLOv11官方开源仓库](https://github.com/ultralytics/ultralytics)的模型和算法进行移植，使之能在SOPHON BM1684/BM1684X/BM1688/CV186X上进行推理测试。

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
建议使用TPU-MLIR编译BModel，在使用TPU-MLIR编译前需要导出ONNX模型。具体可参考[YOLOv11模型导出](./docs/YOLOv11_Export_Guide.md)。

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
│   ├── yolov11s_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684的FP32 BModel，batch_size=1
│   ├── yolov11s_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684的INT8 BModel，batch_size=1
│   ├── yolov11s_int8_4b.bmodel   # 使用TPU-MLIR编译，用于BM1684的INT8 BModel，batch_size=4
│   ├── yolov11s_opt_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684的FP32 BModel，batch_size=1
│   ├── yolov11s_opt_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684的INT8 BModel，batch_size=1
│   └── yolov11s_opt_int8_4b.bmodel   # 使用TPU-MLIR编译，用于BM1684的INT8 BModel，batch_size=4
├── BM1684X
│   ├── yolov11s_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=1
│   ├── yolov11s_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=1
│   ├── yolov11s_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=1
│   ├── yolov11s_int8_4b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=4
│   ├── yolov11s_opt_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=1
│   ├── yolov11s_opt_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=1
│   ├── yolov11s_opt_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=1
│   └── yolov11s_opt_int8_4b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=4
├── BM1688
│   ├── yolov11s_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1, num_core=1
│   ├── yolov11s_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1, num_core=1
│   ├── yolov11s_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=1, num_core=1
│   ├── yolov11s_int8_4b.bmodel   # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=4, num_core=1
│   ├── yolov11s_fp32_1b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1, num_core=2
│   ├── yolov11s_fp16_1b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1, num_core=2
│   ├── yolov11s_int8_1b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=1, num_core=2
│   ├── yolov11s_int8_4b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=4, num_core=2
│   ├── yolov11s_opt_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1, num_core=1
│   ├── yolov11s_opt_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1, num_core=1
│   ├── yolov11s_opt_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=1, num_core=1
│   ├── yolov11s_opt_int8_4b.bmodel   # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=4, num_core=1
│   ├── yolov11s_opt_fp32_1b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1, num_core=2
│   ├── yolov11s_opt_fp16_1b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1, num_core=2
│   ├── yolov11s_opt_int8_1b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=1, num_core=2
│   └── yolov11s_opt_int8_4b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=4, num_core=2
├── CV186X
│   ├── yolov11s_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于CV186X的FP32 BModel，batch_size=1
│   ├── yolov11s_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于CV186X的FP16 BModel，batch_size=1
│   ├── yolov11s_int8_1b.bmodel   # 使用TPU-MLIR编译，用于CV186X的INT8 BModel，batch_size=1
│   ├── yolov11s_int8_4b.bmodel   # 使用TPU-MLIR编译，用于CV186X的INT8 BModel，batch_size=4
│   ├── yolov11s_opt_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于CV186X的FP32 BModel，batch_size=1
│   ├── yolov11s_opt_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于CV186X的FP16 BModel，batch_size=1
│   ├── yolov11s_opt_int8_1b.bmodel   # 使用TPU-MLIR编译，用于CV186X的INT8 BModel，batch_size=1
│   └── yolov11s_opt_int8_4b.bmodel   # 使用TPU-MLIR编译，用于CV186X的INT8 BModel，batch_size=4
└── onnx
    ├── yolov11s.onnx      # 导出的动态onnx模型
    ├── yolov11s_opt.onnx      # 导出的动态opt onnx模型
    ├── yolov11s_qtable_fp16       # TPU-MLIR编译时，用于BM1684X/BM1688的INT8 BModel混合精度量化
    ├── yolov11s_qtable_fp32       # TPU-MLIR编译时，用于BM1684的INT8 BModel混合精度量化
    ├── yolov11s_opt_qtable_fp16       # TPU-MLIR编译时，用于BM1684X/BM1688的INT8 BModel混合精度量化
    └── yolov11s_opt_qtable_fp32       # TPU-MLIR编译时，用于BM1684的INT8 BModel混合精度量化
    
    
         
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

模型编译前需要安装TPU-MLIR，本例程使用的TPU-MLIR版本是`v1.6`，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的SDK中获取)。

- 生成FP32 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684/BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684 #bm1684x/bm1688/cv186x
```

​执行上述命令会在`models/BM1684`下生成`yolov11s_fp32_1b.bmodel`和`yolov11s_opt_fp32_1b.bmodel`文件，即转换好的FP32 BModel。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x #bm1688/cv186x
```

​执行上述命令会在`models/BM1684X/`下生成`yolov11s_fp16_1b.bmodel`和`yolov11s_opt_fp16_1b.bmodel`文件，即转换好的FP16 BModel。

- 生成INT8 BModel

​本例程在`scripts`目录下提供了量化INT8 BModel的脚本，请注意修改`gen_int8bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台（**支持BM1684/BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_int8bmodel_mlir.sh bm1684 #bm1684x/bm1688/cv186x
```

​执行上述命令会在`models/BM1684`下生成`yolov11s_int8_1b.bmodel`和`yolov11s_opt_int8_1b.bmodel`等文件，即转换好的INT8 BModel。量化模型出现问题可以参考：[Calibration_Guide](../../docs/Calibration_Guide.md)。


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
python3 tools/eval_coco.py --gt_path datasets/coco/instances_val2017_1000.json --result_json results/yolov11s_fp32_1b.bmodel_val2017_1000_opencv_python_result.json
```
### 6.2 测试结果
在coco2017 val数据集上，精度测试结果如下：
|   测试平台    |      测试程序     |      测试模型          |AP@IoU=0.5:0.95|AP@IoU=0.5|
| ------------ | ---------------- | ---------------------- | ------------- | -------- |
| SE5-16       | yolov11_opencv.py | yolov11s_fp32_1b.bmodel |    0.471 |    0.639 |
| SE5-16       | yolov11_opencv.py | yolov11s_int8_1b.bmodel |    0.440 |    0.607 |
| SE5-16       | yolov11_opencv.py | yolov11s_int8_4b.bmodel |    0.395 |    0.549 |
| SE5-16       | yolov11_bmcv.py | yolov11s_fp32_1b.bmodel |    0.471 |    0.638 |
| SE5-16       | yolov11_bmcv.py | yolov11s_int8_1b.bmodel |    0.439 |    0.607 |
| SE5-16       | yolov11_bmcv.py | yolov11s_int8_4b.bmodel |    0.398 |    0.551 |
| SE5-16       | yolov11_bmcv.soc | yolov11s_fp32_1b.bmodel |    0.472 |    0.639 |
| SE5-16       | yolov11_bmcv.soc | yolov11s_int8_1b.bmodel |    0.441 |    0.608 |
| SE5-16       | yolov11_bmcv.soc | yolov11s_int8_4b.bmodel |    0.397 |    0.550 |
| SE5-16       | yolov11_bmcv.soc | yolov11s_opt_fp32_1b.bmodel |    0.472 |    0.639 |
| SE5-16       | yolov11_bmcv.soc | yolov11s_opt_int8_1b.bmodel |    0.441 |    0.608 |
| SE5-16       | yolov11_bmcv.soc | yolov11s_opt_int8_4b.bmodel |    0.397 |    0.550 |
| SE7-32       | yolov11_opencv.py | yolov11s_fp32_1b.bmodel |    0.471 |    0.639 |
| SE7-32       | yolov11_opencv.py | yolov11s_fp16_1b.bmodel |    0.471 |    0.639 |
| SE7-32       | yolov11_opencv.py | yolov11s_int8_1b.bmodel |    0.466 |    0.630 |
| SE7-32       | yolov11_opencv.py | yolov11s_int8_4b.bmodel |    0.466 |    0.630 |
| SE7-32       | yolov11_bmcv.py | yolov11s_fp32_1b.bmodel |    0.471 |    0.638 |
| SE7-32       | yolov11_bmcv.py | yolov11s_fp16_1b.bmodel |    0.471 |    0.637 |
| SE7-32       | yolov11_bmcv.py | yolov11s_int8_1b.bmodel |    0.464 |    0.629 |
| SE7-32       | yolov11_bmcv.py | yolov11s_int8_4b.bmodel |    0.464 |    0.629 |
| SE7-32       | yolov11_bmcv.soc | yolov11s_fp32_1b.bmodel |    0.471 |    0.639 |
| SE7-32       | yolov11_bmcv.soc | yolov11s_fp16_1b.bmodel |    0.471 |    0.638 |
| SE7-32       | yolov11_bmcv.soc | yolov11s_int8_1b.bmodel |    0.465 |    0.630 |
| SE7-32       | yolov11_bmcv.soc | yolov11s_int8_4b.bmodel |    0.465 |    0.630 |
| SE7-32       | yolov11_bmcv.soc | yolov11s_opt_fp32_1b.bmodel |    0.471 |    0.639 |
| SE7-32       | yolov11_bmcv.soc | yolov11s_opt_fp16_1b.bmodel |    0.471 |    0.638 |
| SE7-32       | yolov11_bmcv.soc | yolov11s_opt_int8_1b.bmodel |    0.465 |    0.630 |
| SE7-32       | yolov11_bmcv.soc | yolov11s_opt_int8_4b.bmodel |    0.465 |    0.630 |
| SE9-16       | yolov11_opencv.py | yolov11s_fp32_1b.bmodel |    0.471 |    0.639 |
| SE9-16       | yolov11_opencv.py | yolov11s_fp16_1b.bmodel |    0.471 |    0.639 |
| SE9-16       | yolov11_opencv.py | yolov11s_int8_1b.bmodel |    0.466 |    0.630 |
| SE9-16       | yolov11_opencv.py | yolov11s_int8_4b.bmodel |    0.466 |    0.630 |
| SE9-16       | yolov11_bmcv.py | yolov11s_fp32_1b.bmodel |    0.471 |    0.638 |
| SE9-16       | yolov11_bmcv.py | yolov11s_fp16_1b.bmodel |    0.471 |    0.638 |
| SE9-16       | yolov11_bmcv.py | yolov11s_int8_1b.bmodel |    0.462 |    0.625 |
| SE9-16       | yolov11_bmcv.py | yolov11s_int8_4b.bmodel |    0.462 |    0.625 |
| SE9-16       | yolov11_bmcv.soc | yolov11s_fp32_1b.bmodel |    0.472 |    0.639 |
| SE9-16       | yolov11_bmcv.soc | yolov11s_fp16_1b.bmodel |    0.472 |    0.638 |
| SE9-16       | yolov11_bmcv.soc | yolov11s_int8_1b.bmodel |    0.463 |    0.625 |
| SE9-16       | yolov11_bmcv.soc | yolov11s_int8_4b.bmodel |    0.463 |    0.625 |
| SE9-16       | yolov11_opencv.py | yolov11s_fp32_1b_2core.bmodel |    0.471 |    0.639 |
| SE9-16       | yolov11_opencv.py | yolov11s_fp16_1b_2core.bmodel |    0.471 |    0.639 |
| SE9-16       | yolov11_opencv.py | yolov11s_int8_1b_2core.bmodel |    0.466 |    0.630 |
| SE9-16       | yolov11_opencv.py | yolov11s_int8_4b_2core.bmodel |    0.466 |    0.630 |
| SE9-16       | yolov11_bmcv.py | yolov11s_fp32_1b_2core.bmodel |    0.471 |    0.638 |
| SE9-16       | yolov11_bmcv.py | yolov11s_fp16_1b_2core.bmodel |    0.471 |    0.638 |
| SE9-16       | yolov11_bmcv.py | yolov11s_int8_1b_2core.bmodel |    0.462 |    0.625 |
| SE9-16       | yolov11_bmcv.py | yolov11s_int8_4b_2core.bmodel |    0.462 |    0.625 |
| SE9-16       | yolov11_bmcv.soc | yolov11s_fp32_1b_2core.bmodel |    0.472 |    0.639 |
| SE9-16       | yolov11_bmcv.soc | yolov11s_fp16_1b_2core.bmodel |    0.472 |    0.638 |
| SE9-16       | yolov11_bmcv.soc | yolov11s_int8_1b_2core.bmodel |    0.463 |    0.625 |
| SE9-16       | yolov11_bmcv.soc | yolov11s_int8_4b_2core.bmodel |    0.463 |    0.625 |
| SE9-16       | yolov11_bmcv.soc | yolov11s_opt_fp32_1b.bmodel |    0.472 |    0.639 |
| SE9-16       | yolov11_bmcv.soc | yolov11s_opt_fp16_1b.bmodel |    0.472 |    0.638 |
| SE9-16       | yolov11_bmcv.soc | yolov11s_opt_int8_1b.bmodel |    0.463 |    0.625 |
| SE9-16       | yolov11_bmcv.soc | yolov11s_opt_int8_4b.bmodel |    0.463 |    0.625 |
| SE9-16       | yolov11_bmcv.soc | yolov11s_opt_fp32_1b_2core.bmodel |    0.472 |    0.639 |
| SE9-16       | yolov11_bmcv.soc | yolov11s_opt_fp16_1b_2core.bmodel |    0.472 |    0.638 |
| SE9-16       | yolov11_bmcv.soc | yolov11s_opt_int8_1b_2core.bmodel |    0.463 |    0.625 |
| SE9-16       | yolov11_bmcv.soc | yolov11s_opt_int8_4b_2core.bmodel |    0.463 |    0.625 |
| SE9-8        | yolov11_opencv.py | yolov11s_fp32_1b.bmodel |    0.471 |    0.639 |
| SE9-8        | yolov11_opencv.py | yolov11s_fp16_1b.bmodel |    0.471 |    0.639 |
| SE9-8        | yolov11_opencv.py | yolov11s_int8_1b.bmodel |    0.466 |    0.630 |
| SE9-8        | yolov11_opencv.py | yolov11s_int8_4b.bmodel |    0.466 |    0.630 |
| SE9-8        | yolov11_bmcv.py | yolov11s_fp32_1b.bmodel |    0.471 |    0.638 |
| SE9-8        | yolov11_bmcv.py | yolov11s_fp16_1b.bmodel |    0.471 |    0.638 |
| SE9-8        | yolov11_bmcv.py | yolov11s_int8_1b.bmodel |    0.462 |    0.625 |
| SE9-8        | yolov11_bmcv.py | yolov11s_int8_4b.bmodel |    0.462 |    0.625 |
| SE9-8        | yolov11_bmcv.soc | yolov11s_fp32_1b.bmodel |    0.472 |    0.639 |
| SE9-8        | yolov11_bmcv.soc | yolov11s_fp16_1b.bmodel |    0.472 |    0.638 |
| SE9-8        | yolov11_bmcv.soc | yolov11s_int8_1b.bmodel |    0.463 |    0.625 |
| SE9-8        | yolov11_bmcv.soc | yolov11s_int8_4b.bmodel |    0.463 |    0.625 |
| SE9-8        | yolov11_bmcv.soc | yolov11s_opt_fp32_1b.bmodel |    0.472 |    0.639 |
| SE9-8        | yolov11_bmcv.soc | yolov11s_opt_fp16_1b.bmodel |    0.472 |    0.638 |
| SE9-8        | yolov11_bmcv.soc | yolov11s_opt_int8_1b.bmodel |    0.463 |    0.625 |
| SE9-8        | yolov11_bmcv.soc | yolov11s_opt_int8_4b.bmodel |    0.463 |    0.625 |

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
bmrt_test --bmodel models/BM1684/yolov11s_fp32_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

|              测试模型               | calculate time(ms) |
| ----------------------------------- | ----------------- |
| BM1684/yolov11s_fp32_1b.bmodel     |          27.02  |
| BM1684/yolov11s_int8_1b.bmodel     |          18.68  |
| BM1684/yolov11s_int8_4b.bmodel     |          11.20  |
| BM1684/yolov11s_opt_fp32_1b.bmodel |          27.25  |
| BM1684/yolov11s_opt_int8_1b.bmodel |          18.91  |
| BM1684/yolov11s_opt_int8_4b.bmodel |          11.43  |
| BM1684X/yolov11s_fp32_1b.bmodel    |          24.66  |
| BM1684X/yolov11s_fp16_1b.bmodel    |           5.85  |
| BM1684X/yolov11s_int8_1b.bmodel    |           3.39  |
| BM1684X/yolov11s_int8_4b.bmodel    |           3.07  |
| BM1684X/yolov11s_opt_fp32_1b.bmodel|          24.82  |
| BM1684X/yolov11s_opt_fp16_1b.bmodel|           5.99  |
| BM1684X/yolov11s_opt_int8_1b.bmodel|           3.54  |
| BM1684X/yolov11s_opt_int8_4b.bmodel|           3.21  |
| BM1688/yolov11s_fp32_1b.bmodel     |         131.47  |
| BM1688/yolov11s_fp16_1b.bmodel     |          33.56  |
| BM1688/yolov11s_int8_1b.bmodel     |           8.19  |
| BM1688/yolov11s_int8_4b.bmodel     |           7.75  |
| BM1688/yolov11s_fp32_1b_2core.bmodel|          70.09  |
| BM1688/yolov11s_fp16_1b_2core.bmodel|          19.38  |
| BM1688/yolov11s_int8_1b_2core.bmodel|           6.51  |
| BM1688/yolov11s_int8_4b_2core.bmodel|           5.23  |
| BM1688/yolov11s_opt_fp32_1b.bmodel |         131.76  |
| BM1688/yolov11s_opt_fp16_1b.bmodel |          33.66  |
| BM1688/yolov11s_opt_int8_1b.bmodel |           8.27  |
| BM1688/yolov11s_opt_int8_4b.bmodel |           7.97  |
| BM1688/yolov11s_opt_fp32_1b_2core.bmodel|          70.40  |
| BM1688/yolov11s_opt_fp16_1b_2core.bmodel|          19.50  |
| BM1688/yolov11s_opt_int8_1b_2core.bmodel|           6.61  |
| BM1688/yolov11s_opt_int8_4b_2core.bmodel|           5.41  |
| CV186X/yolov11s_fp32_1b.bmodel     |         131.50  |
| CV186X/yolov11s_fp16_1b.bmodel     |          33.56  |
| CV186X/yolov11s_int8_1b.bmodel     |           8.21  |
| CV186X/yolov11s_int8_4b.bmodel     |           7.73  |
| CV186X/yolov11s_opt_fp32_1b.bmodel |         131.82  |
| CV186X/yolov11s_opt_fp16_1b.bmodel |          33.66  |
| CV186X/yolov11s_opt_int8_1b.bmodel |           8.32  |
| CV186X/yolov11s_opt_int8_4b.bmodel |           7.95  |

> **测试说明**：  
1. 性能测试结果具有一定的波动性；
2. `calculate time`已折算为平均每张图片的推理时间；
3. SoC和PCIe的测试结果基本一致。


### 7.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++和Python例程打印的时间已经折算为单张图片的处理时间。

在不同的测试平台上，使用不同的例程、模型测试`datasets/val2017_1000`，conf_thresh=0.25，nms_thresh=0.7，性能测试结果如下：
|    测试平台  |     测试程序      |        测试模型        |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ---------------- | ---------------------- | --------  | ---------    | ---------     | ---------      |
|   SE5-16    | yolov11_opencv.py | yolov11s_fp32_1b.bmodel |      9.96       |      21.93      |      32.12      |      5.04       |
|   SE5-16    | yolov11_opencv.py | yolov11s_int8_1b.bmodel |      6.74       |      21.51      |      23.76      |      4.99       |
|   SE5-16    | yolov11_opencv.py | yolov11s_int8_4b.bmodel |      6.89       |      23.83      |      16.13      |      4.94       |
|   SE5-16    |  yolov11_bmcv.py  | yolov11s_fp32_1b.bmodel |      3.63       |      2.76       |      29.34      |      4.95       |
|   SE5-16    |  yolov11_bmcv.py  | yolov11s_int8_1b.bmodel |      3.62       |      2.75       |      20.94      |      4.91       |
|   SE5-16    |  yolov11_bmcv.py  | yolov11s_int8_4b.bmodel |      3.47       |      2.59       |      13.10      |      4.35       |
|   SE5-16    | yolov11_bmcv.soc  | yolov11s_fp32_1b.bmodel |      4.91       |      1.56       |      26.93      |      8.52       |
|   SE5-16    | yolov11_bmcv.soc  | yolov11s_int8_1b.bmodel |      4.90       |      1.55       |      18.59      |      8.53       |
|   SE5-16    | yolov11_bmcv.soc  | yolov11s_int8_4b.bmodel |      4.75       |      1.49       |      11.18      |      8.49       |
|   SE5-16    | yolov11_bmcv.soc  |yolov11s_opt_fp32_1b.bmodel|      4.89       |      1.55       |      27.16      |      2.60       |
|   SE5-16    | yolov11_bmcv.soc  |yolov11s_opt_int8_1b.bmodel|      4.90       |      1.56       |      18.82      |      2.60       |
|   SE5-16    | yolov11_bmcv.soc  |yolov11s_opt_int8_4b.bmodel|      4.74       |      1.49       |      11.40      |      2.67       |
|   SE7-32    | yolov11_opencv.py | yolov11s_fp32_1b.bmodel |      6.80       |      22.42      |      30.43      |      5.40       |
|   SE7-32    | yolov11_opencv.py | yolov11s_fp16_1b.bmodel |      6.82       |      22.88      |      11.69      |      5.39       |
|   SE7-32    | yolov11_opencv.py | yolov11s_int8_1b.bmodel |      6.81       |      22.52      |      9.19       |      5.33       |
|   SE7-32    | yolov11_opencv.py | yolov11s_int8_4b.bmodel |      6.82       |      25.31      |      8.73       |      5.39       |
|   SE7-32    |  yolov11_bmcv.py  | yolov11s_fp32_1b.bmodel |      3.18       |      2.34       |      27.19      |      5.46       |
|   SE7-32    |  yolov11_bmcv.py  | yolov11s_fp16_1b.bmodel |      3.15       |      2.34       |      8.31       |      5.52       |
|   SE7-32    |  yolov11_bmcv.py  | yolov11s_int8_1b.bmodel |      3.15       |      2.33       |      5.86       |      5.41       |
|   SE7-32    |  yolov11_bmcv.py  | yolov11s_int8_4b.bmodel |      2.99       |      2.15       |      5.18       |      4.81       |
|   SE7-32    | yolov11_bmcv.soc  | yolov11s_fp32_1b.bmodel |      4.36       |      0.75       |      24.48      |      8.93       |
|   SE7-32    | yolov11_bmcv.soc  | yolov11s_fp16_1b.bmodel |      4.37       |      0.75       |      5.68       |      8.96       |
|   SE7-32    | yolov11_bmcv.soc  | yolov11s_int8_1b.bmodel |      4.37       |      0.75       |      3.21       |      8.96       |
|   SE7-32    | yolov11_bmcv.soc  | yolov11s_int8_4b.bmodel |      4.20       |      0.72       |      3.01       |      8.94       |
|   SE7-32    | yolov11_bmcv.soc  |yolov11s_opt_fp32_1b.bmodel|      4.35       |      0.75       |      24.63      |      2.62       |
|   SE7-32    | yolov11_bmcv.soc  |yolov11s_opt_fp16_1b.bmodel|      4.35       |      0.75       |      5.84       |      2.62       |
|   SE7-32    | yolov11_bmcv.soc  |yolov11s_opt_int8_1b.bmodel|      4.39       |      0.75       |      3.40       |      2.65       |
|   SE7-32    | yolov11_bmcv.soc  |yolov11s_opt_int8_4b.bmodel|      4.18       |      0.72       |      3.17       |      2.69       |
|   SE9-16    | yolov11_opencv.py | yolov11s_fp32_1b.bmodel |      16.62      |      30.03      |     138.75      |      6.94       |
|   SE9-16    | yolov11_opencv.py | yolov11s_fp16_1b.bmodel |      9.48       |      29.50      |      40.82      |      6.93       |
|   SE9-16    | yolov11_opencv.py | yolov11s_int8_1b.bmodel |      9.45       |      29.94      |      15.80      |      6.81       |
|   SE9-16    | yolov11_opencv.py | yolov11s_int8_4b.bmodel |      9.47       |      33.37      |      14.88      |      7.36       |
|   SE9-16    |  yolov11_bmcv.py  | yolov11s_fp32_1b.bmodel |      4.02       |      4.44       |     134.92      |      7.14       |
|   SE9-16    |  yolov11_bmcv.py  | yolov11s_fp16_1b.bmodel |      3.99       |      4.40       |      37.02      |      7.01       |
|   SE9-16    |  yolov11_bmcv.py  | yolov11s_int8_1b.bmodel |      3.98       |      4.39       |      11.62      |      6.90       |
|   SE9-16    |  yolov11_bmcv.py  | yolov11s_int8_4b.bmodel |      3.81       |      4.11       |      10.44      |      6.08       |
|   SE9-16    | yolov11_bmcv.soc  | yolov11s_fp32_1b.bmodel |      5.50       |      1.73       |     131.40      |      12.15      |
|   SE9-16    | yolov11_bmcv.soc  | yolov11s_fp16_1b.bmodel |      5.54       |      1.72       |      33.48      |      12.14      |
|   SE9-16    | yolov11_bmcv.soc  | yolov11s_int8_1b.bmodel |      5.53       |      1.72       |      8.10       |      12.16      |
|   SE9-16    | yolov11_bmcv.soc  | yolov11s_int8_4b.bmodel |      5.37       |      1.64       |      7.74       |      12.09      |
|   SE9-16    | yolov11_opencv.py |yolov11s_fp32_1b_2core.bmodel|      9.46       |      29.96      |      77.64      |      6.93       |
|   SE9-16    | yolov11_opencv.py |yolov11s_fp16_1b_2core.bmodel|      9.43       |      29.84      |      26.66      |      6.93       |
|   SE9-16    | yolov11_opencv.py |yolov11s_int8_1b_2core.bmodel|      9.39       |      29.42      |      13.73      |      6.81       |
|   SE9-16    | yolov11_opencv.py |yolov11s_int8_4b_2core.bmodel|      9.52       |      33.06      |      12.35      |      6.98       |
|   SE9-16    |  yolov11_bmcv.py  |yolov11s_fp32_1b_2core.bmodel|      4.02       |      4.41       |      73.46      |      7.02       |
|   SE9-16    |  yolov11_bmcv.py  |yolov11s_fp16_1b_2core.bmodel|      3.98       |      4.42       |      22.77      |      7.00       |
|   SE9-16    |  yolov11_bmcv.py  |yolov11s_int8_1b_2core.bmodel|      3.99       |      4.40       |      9.86       |      6.91       |
|   SE9-16    |  yolov11_bmcv.py  |yolov11s_int8_4b_2core.bmodel|      3.81       |      4.09       |      7.92       |      6.05       |
|   SE9-16    | yolov11_bmcv.soc  |yolov11s_fp32_1b_2core.bmodel|      5.55       |      1.72       |      70.05      |      12.15      |
|   SE9-16    | yolov11_bmcv.soc  |yolov11s_fp16_1b_2core.bmodel|      5.56       |      1.72       |      19.32      |      12.15      |
|   SE9-16    | yolov11_bmcv.soc  |yolov11s_int8_1b_2core.bmodel|      5.54       |      1.72       |      6.44       |      12.16      |
|   SE9-16    | yolov11_bmcv.soc  |yolov11s_int8_4b_2core.bmodel|      5.34       |      1.64       |      5.21       |      12.10      |
|   SE9-16    | yolov11_bmcv.soc  |yolov11s_opt_fp32_1b.bmodel|      5.53       |      1.73       |     131.69      |      3.68       |
|   SE9-16    | yolov11_bmcv.soc  |yolov11s_opt_fp16_1b.bmodel|      5.53       |      1.73       |      33.58      |      3.66       |
|   SE9-16    | yolov11_bmcv.soc  |yolov11s_opt_int8_1b.bmodel|      5.52       |      1.73       |      8.21       |      3.78       |
|   SE9-16    | yolov11_bmcv.soc  |yolov11s_opt_int8_4b.bmodel|      5.35       |      1.64       |      7.96       |      3.69       |
|   SE9-16    | yolov11_bmcv.soc  |yolov11s_opt_fp32_1b_2core.bmodel|      5.58       |      1.72       |      70.34      |      4.04       |
|   SE9-16    | yolov11_bmcv.soc  |yolov11s_opt_fp16_1b_2core.bmodel|      5.53       |      1.72       |      19.42      |      3.66       |
|   SE9-16    | yolov11_bmcv.soc  |yolov11s_opt_int8_1b_2core.bmodel|      5.57       |      1.73       |      6.55       |      3.66       |
|   SE9-16    | yolov11_bmcv.soc  |yolov11s_opt_int8_4b_2core.bmodel|      5.38       |      1.64       |      5.42       |      3.69       |
|    SE9-8    | yolov11_opencv.py | yolov11s_fp32_1b.bmodel |      9.48       |      30.13      |     138.92      |      7.01       |
|    SE9-8    | yolov11_opencv.py | yolov11s_fp16_1b.bmodel |      9.44       |      30.14      |      40.84      |      7.02       |
|    SE9-8    | yolov11_opencv.py | yolov11s_int8_1b.bmodel |      9.44       |      29.67      |      15.64      |      6.88       |
|    SE9-8    | yolov11_opencv.py | yolov11s_int8_4b.bmodel |      9.50       |      33.73      |      15.01      |      7.50       |
|    SE9-8    |  yolov11_bmcv.py  | yolov11s_fp32_1b.bmodel |      4.08       |      4.44       |     135.05      |      7.09       |
|    SE9-8    |  yolov11_bmcv.py  | yolov11s_fp16_1b.bmodel |      4.04       |      4.41       |      36.88      |      7.07       |
|    SE9-8    |  yolov11_bmcv.py  | yolov11s_int8_1b.bmodel |      4.04       |      4.40       |      11.43      |      6.96       |
|    SE9-8    |  yolov11_bmcv.py  | yolov11s_int8_4b.bmodel |      3.85       |      4.11       |      10.41      |      6.32       |
|    SE9-8    | yolov11_bmcv.soc  | yolov11s_fp32_1b.bmodel |      5.54       |      1.73       |     131.39      |      12.22      |
|    SE9-8    | yolov11_bmcv.soc  | yolov11s_fp16_1b.bmodel |      5.61       |      1.73       |      33.44      |      12.22      |
|    SE9-8    | yolov11_bmcv.soc  | yolov11s_int8_1b.bmodel |      5.59       |      1.73       |      8.04       |      12.23      |
|    SE9-8    | yolov11_bmcv.soc  | yolov11s_int8_4b.bmodel |      5.39       |      1.64       |      7.70       |      12.16      |
|    SE9-8    | yolov11_bmcv.soc  |yolov11s_opt_fp32_1b.bmodel|      5.57       |      1.73       |     131.68      |      3.69       |
|    SE9-8    | yolov11_bmcv.soc  |yolov11s_opt_fp16_1b.bmodel|      5.62       |      1.73       |      33.53      |      3.66       |
|    SE9-8    | yolov11_bmcv.soc  |yolov11s_opt_int8_1b.bmodel|      5.63       |      1.72       |      8.14       |      3.68       |
|    SE9-8    | yolov11_bmcv.soc  |yolov11s_opt_int8_4b.bmodel|      5.43       |      1.64       |      7.93       |      3.71       |

> **测试说明**：  
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. SE5-16/SE7-32的主控处理器均为8核CA53@2.3GHz，SE9-16为8核CA53@1.6GHz，SE9-8为6核CA53@1.6GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异，不同的阈值对后处理时间影响较大。 


## 8. FAQ
请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。