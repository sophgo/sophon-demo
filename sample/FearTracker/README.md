# FearTracker

## 目录

- [FearTracker](#feartracker)
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
FEAR（Fast, Efficient, Accurate and Robust Visual Tracker）是一个基于深度学习的单目标视觉跟踪模型，出自ECCV 2022。模型采用FBNet-C作为backbone，使用Siamese跟踪架构，通过交叉相关（cross-correlation）在搜索图像中定位目标，具有参数量小、推理速度快的特点。

本例程对[FEARTracker](https://github.com/vasyl-borsuk/FEARTracker)的预训练模型进行移植，导出为ONNX模型，并编译为BModel使之能在SOPHON BM1684X和BM1688上进行推理测试。

## 2. 特性
* 支持BM1684X(x86 PCIe, SoC)、BM1688(SoC)
* 支持FP16模型编译和推理
* 支持基于sail的Python推理
* 支持视频文件的单目标跟踪
* 支持自定义初始边界框、输出视频保存

## 3. 准备模型与数据
建议使用TPU-MLIR编译BModel。原始PyTorch模型需要先导出为ONNX格式，具体可参考[FEARTracker模型导出](./docs/FearTracker_Export_Guide.md)。

本例程在`tools`目录下提供了模型导出脚本`export_onnx.py`。

同时，您需要准备用于测试的视频文件和初始边界框（x,y,w,h格式）。

通过运行如下脚本，可以拷贝本例程所需的数据和模型（从源码项目复制已编译的bmodel）：
```bash
bash scripts/download.sh --all
```

下载的模型包括：
```
./models
├── onnx #onnx文件
├── BM1684X
│   ├── feartracker_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=1
│   └── feartracker_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=1
└── BM1688
    └── feartracker_bm1688_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1
```

测试数据包括：
```
./datasets
└── test.mp4                              # 测试用视频文件
```

## 4. 模型编译
导出的ONNX模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。建议使用TPU-MLIR编译BModel。

模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的"3. 编译ONNX模型"(请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的SDK中获取)。

- 生成FP16 BModel

本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688**），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x
```

执行上述命令会在`models/BM1684X/`下生成`feartracker_fp16_1b.bmodel`文件，即转换好的FP16 BModel。

注意：模型有2个输入（template=[1,3,128,128]模板图像, search=[1,3,256,256]搜索图像）和2个输出（bbox_pred=[1,4,16,16]回归预测, cls_pred=[1,1,16,16]分类预测），编译时必须指定`--channel_format none`（非图片模型）。

## 5. 例程测试
- [Python例程](./python/README.md)

## 6. 精度测试
### 6.1 测试方法

精度验证通过对比TPU推理结果与PyTorch参考模型的逐帧跟踪结果来完成。在相同的初始边界框下，比较两者的跟踪轨迹（每帧预测的边界框）。

### 6.2 测试结果
待补充。

## 7. 性能测试
### 7.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684X/feartracker_fp32_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间。

待补充。

### 7.2 程序运行性能
参考[Python例程](python/README.md)运行程序，并查看统计的推理时间。

在不同的测试平台上，使用不同的模型进行测试，性能测试结果如下：

|    测试平台  |     测试程序      |             测试模型                     |  帧率   | 平均推理时间(ms) |
| ----------- | ---------------- | --------------------------------------- | ------- | ---------------- |
|   SE9-16    | fear_tracker.py  | BM1688/feartracker_bm1688_fp16_1b.bmodel|   64    |      15.5        |

> **测试说明**：
> 1. 测试视频：661帧，分辨率 640x360；
> 2. 平均推理时间为完整单帧处理时间，包括模板/搜索图像裁剪、numpy预处理、TPU推理（SYSIO模式）、后处理（sigmoid→argmax→bbox解码→坐标缩放）；
> 3. 模板图像仅在首帧预处理一次，后续帧复用模板图像，不计入后续帧的推理时间；
> 4. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 5. SE9-16的主控处理器为8核CA55@1.6GHz，BM1688 TPU；

## 8. FAQ
1. **推理结果与参考模型不一致**: 可检查初始边界框是否一致，以及模型是否使用相同的配置参数（template_size=128, instance_size=256, score_size=16）。

其他常见问题请参考[SOPHON-DEMO FAQ](../../docs/FAQ.md)。