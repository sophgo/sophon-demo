[简体中文](./README.md)

# GroundingDINO

## 目录

* [1. 简介](#1-简介)
* [2. 特性](#2-特性)
* [3. 准备模型与数据](#3-准备模型与数据)
* [4. 模型编译](#4-模型编译)
  * [4.1 导出onnx模型](#4.1-导出onnx模型)
  * [4.2 导出bmodel模型](#4.2-导出bmodel模型)
* [5. 例程测试](#5-例程测试)
* [6. 性能测试](#7-性能测试)
  * [6.1 bmrt_test](#6.1-bmrt_test)
  * [6.2 程序运行性能](#6.2-程序运行性能)
* [7. FAQ](#7-faq)

## 1. 简介
GroundingDINO是一种多模态的目标检测模型。
本例程对[GroundingDINO官方开源仓库](https://github.com/IDEA-Research/GroundingDINO/tree/main)的模型和算法进行移植，使之能在SOPHON BM1684和BM1684X上进行推理测试,移植过程中针对TPU的推理上对源代码进行了优化和提速。

## 2. 特性
* 支持BM1684X(x86 PCIe、SoC)
* 支持FP16模型编译和推理
* 支持基于PIL的Python推理
* 支持单batch模型推理
* 支持图片测试
 
## 3. 准备模型与数据
建议使用TPU-MLIR编译BModel，Pytorch模型在编译前要导出成onnx模型，如果您使用的tpu-mlir版本>=v1.4.0（即官网v23.09.01）。其中Pytorch转onnx模型具体可参考[常见问题](./docs/GroundingDINO_Common_Problems.md)。

​本例程在`scripts`目录下提供了相关模型和数据的下载脚本`download.sh`，您也可以自己准备模型和数据集，并参考[4. 模型编译](#4-模型编译)进行模型转换。

```bash
# 安装unzip，若已安装请跳过，非ubuntu系统视情况使用yum或其他方式安装
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```

执行下载脚本后，当前目录下的文件如下：
```bash
├── docs
│   └── GroundingDINO_Common_Problems.md        #GroundingDINO 常见问题及解答
├── models
│   ├── bert-base-uncased                       # tokenizer 分词器文件夹					
│   ├── BM1684X
│   │  └── groundingdino_bm1684x_fp16.bmodel    # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=1
│   ├── torch
│   │   └── groundingdino_swint_ogc.pth	     # pytorch模型
│   └── onnx
│       └── GroundingDino.onnx             	 # 导出的onnx动态模型
├── datasets
│   ├── test                                      # 测试图片
│   ├── test_car_person_1080P.mp4                 # 测试视频
│   ├── coco.names                                # coco类别名文件
│   ├── coco128                                   # coco128数据集，用于模型量化
│   └── coco                                      # coco数据集
│       ├── val2017_1000                          # coco val2017_1000数据集：coco val2017中随机抽取的1000张样本
│       └── instances_val2017_1000.json           # coco val2017_1000数据集标签文件，用于计算精度评价指标  
├── python
│   ├── PostProcess.py                     #后处理脚本
│   ├── README.md                          #python例程执行指南
│   ├── groundingdino_pil.py               #GroundingDINO推理脚本
│   ├── requirements.txt                   #python例程的依赖模块
│   └── utils.py                           #辅助函数文件
├── scripts                         
│   ├── download.sh                        #下载脚本
│   └── gen_fp16bmodel_mlir.sh             #模型编译脚本
└── README.md                              #GroundingDINO例程指南
```

## 4. 模型编译
导出的模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。建议使用TPU-MLIR编译BModel。

模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index/material/31/all.html)相应版本的SDK中获取)。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X**），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x
```

​执行上述命令会在`models/BM1684X`下生成`groundingdino_bm1684x_fp16.bmodel` 即用于推理的FP16 BModel。

## 5. 例程测试
目前提供python版本的例程，请参考:
- [Python例程](./python/README.md)

## 6. 性能测试
### 6.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684X/groundingdino_bm1684x_fp16.bmodel
```
测试结果中的`calculate time`就是模型推理的时间。

测试各个模型的理论推理时间，结果如下：

|              测试模型                | calculate time(s)         |
| ------------------------------------| --------------------------|
| groundingdino_bm1684x_fp16.bmodel   | 0.532807                  |

> **测试说明**：  
> 1. 性能测试结果具有一定的波动性；
> 2. `calculate time`已折算为平均每张图片的推理时间；
> 3. SoC和PCIe的测试结果基本一致。

### 6.2 程序运行性能
参考[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。目前GroundingDINO仅支持1 batch的fp16模型。

测试`datasets/test/zidane.jpg`单张图片性能测试结果如下（时间单位为ms），测试结果有一定波动性：
| 测试平台     |       测试程序         |               测试模型             | decode_time | preprocess_time | inference_time  |postprocess_time | 
| ----------- | ------------------   | --------------------------------- | ----------- | --------------- | --------------- | ---------------- |
| BM1684X SoC | groundingdino_pil.py | groundingdino_bm1684x_fp16.bmodel | 3.50        | 36.25           | 547.12          | 2.73                |

> **测试说明**：  
> 1. 时间单位均为毫秒(s)；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. BM1684/1684X SoC的主控处理器均为8核 ARM A53 42320 DMIPS @2.3GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异，不同的阈值对后处理时间影响较大。 

## 7. FAQ
GroundingDINO移植相关问题可参考[GroundingDINO常见问题](./docs/GroundingDINO_Common_Problems.md)，其他问题请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。