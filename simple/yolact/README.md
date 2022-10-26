# YOLACT

## 目录

* [YOLACT](#YOLACT)
  * [目录](#目录)
  * [1. 简介](#1-简介)
  * [2. 数据集](#2-数据集)
  * [3. 准备环境与数据](#3-准备环境与数据)
  * [4. 模型转换](#4-模型转换)
    * [4.1 生成FP32 BModel](#41-生成fp32-bmodel)
  * [5. 例程测试](#5-例程测试)

## 1. 简介

YOLACT是一种实时的实例分割的方法。

论文地址: [YOLACT: Real-time Instance Segmentation](https://arxiv.org/abs/1904.02689)

官方源码: https://github.com/dbolya/yolact

![](./pics/yolact_example_0.png)

## 2. 数据集

[MS COCO](http://cocodataset.org/#home)，是微软构建的一个包含分类、检测、分割等任务的大型的数据集。使用[yolact](https://github.com/dbolya/yolact)基于COCO Detection 2017预训练好的80类通用目标检测模型。

> MS COCO提供了一些[API](https://github.com/cocodataset/cocoapi)，方便对数据集的使用和模型评估，您可以使用pip安装` pip3 install pycocotools`，并使用COCO提供的API进行下载。

## 3 准备环境与数据

Pytorch的模型在编译前要经过`torch.jit.trace`，trace后的模型才能用于编译BModel。trace的方法和原理可参考[torch.jit.trace参考文档](../docs/torch.jit.trace_Guide.md)。

同时，您需要准备用于测试的数据。

本例程在`${yolact}/scripts`目录下提供了相关模型和数据集的下载脚本，您也可以自己准备模型和数据集，并参考[4. 模型编译](#4-模型编译)进行模型转换。

```bash
chmod +x ./*
./scripts/00_prepare_test_data.sh
./scripts/01_prepare_model.sh
```

执行后，模型保存至`${yolact}/data/models`，图片数据集下载并解压至`${yolact}/data/images/`，视频数据集下载并解压至`${yolact}/data/videos/`

```bash
data
├── images											# 测试图像数据文件夹
│   ├── 000000162415.jpg
│   ├── 000000250758.jpg
│   ├── 000000404484.jpg
│   ├── 000000404568.jpg
│   ├── n02412080_66.JPEG
│   └── n07697537_55793.JPEG
├── models
│   ├── BM1684										#
│   │   ├── yolact_base_54_800000_fp32_1b.bmodel
│   │   └── yolact_base_54_800000_fp32_4b.bmodel
│   ├── BM1684X
│   │   ├── yolact_base_54_800000_fp32_1b.bmodel
│   │   └── yolact_base_54_800000_fp32_4b.bmodel
│   └── torch
│       ├── yolact_base_54_800000.pth				# 官方原始模型
│       └── yolact_base_54_800000.trace.pt			# trace后的模型
└── videos											# 测试视频数据文件夹
    └── road.mp4
```

## 4. 模型编译

trace后的pytorch模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。

模型编译前需要安装TPU-NNTC(>=3.1.0)，具体可参考[tpu-nntc环境搭建](../docs/Environment_Install_Guide.md#1-tpu-nntc环境搭建)。

下面我们以`yolact_base_54_800000`模型为例，介绍如何完成模型的编译。

### 4.1 生成FP32 BModel

pytorch模型编译为FP32 BModel，具体方法可参考《TPU-NNTC开发参考手册》的`BMNETP使用`章节。注：需要获取《TPU-NNTC开发参考手册》，请联系技术支持。

本例程在`scripts`目录下提供了编译FP32 BModel的脚本。请注意修改`10_gen_fp32bmodel.sh`中的JIT模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（支持BM1684和BM1684X），如：

执行以下命令，使用bmnetp编译生成FP32 BModel：

```bash
# 编译BM1684模型： ./scripts/10_gen_fp32bmodel.sh BM1684
# 编译BM1684X模型： ./scripts/10_gen_fp32bmodel.sh BM1684X
chmod +x ./*
./scripts/10_gen_fp32bmodel.sh BM1684X
```

执行上述命令会在`${yolact}/data/models/BM1684X/`下生成`yolact_base_54_800000_fp32_1b.bmodel、yolact_base_54_800000_fp32_4b.bmodel`文件，即转换好的1684X FP32 BModel。

## 5. 例程测试

- [Python例程](python/README.md)
