# Resnet

## 目录

- [Resnet](#resnet)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 数据集](#2-数据集)
  - [3. 准备模型与数据](#3-准备模型与数据)
  - [4. 模型编译](#4-模型编译)
    - [4.1 生成FP32 BModel](#41-生成fp32-bmodel)
    - [4.2 生成INT8 BModel](#42-生成int8-bmodel)
  - [5. 例程测试](#5-例程测试)
    


## 1. 简介

  本例程对[torchvision Resnet](https://pytorch.org/vision/stable/models.html)的模型和算法进行移植，使之能在SOPHON BM1684和BM1684X上进行推理测试。

**论文:** [Resnet论文](https://arxiv.org/abs/1512.03385)

深度残差网络（Deep residual network, ResNet）是由于Kaiming He等在2015提出的深度神经网络结构，它利用残差学习来解决深度神经网络训练退化的问题。

在此非常感谢Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun等人的贡献。

## 2. 数据集

[ImageNet](https://www.image-net.org)验证集`ILSVRC2012_img_val`中随机抽取了1000张图片，作为本例程测试数据集。

[ImageNet](https://www.image-net.org)验证集`ILSVRC2012_img_val`中随机抽取了200张图片，作为本例程量化数据集。


## 3. 准备模型与数据
Pytorch的模型在编译前要经过`torch.jit.trace`，trace后的模型才能用于编译BModel。trace的方法和原理可参考[torch.jit.trace参考文档](../../docs/torch.jit.trace_Guide.md)。

同时，您需要准备用于测试的数据集，如果量化模型，还要准备用于量化的数据集。

本例程在`scripts`目录下提供了相关模型和数据集的下载脚本`download.sh`，您也可以自己准备模型和数据集，并参考[4. 模型编译](#4-模型编译)进行模型转换。
```bash
chmod +x ./scripts/*
./scripts/download.sh
```
执行后，模型保存至`data/models`，测试数据集下载并解压至`data/images/imagenet_val_1k`，量化数据集下载并解压至`data/images/cali_data`

下载的模型包括：
```
.
├── BM1684
│   ├── resnet_fp32_b1.bmodel 用于BM1684的FP32 BModel，batch_size=1
│   ├── resnet_fp32_b4.bmodel 用于BM1684的FP32 BModel，batch_size=4
│   ├── resnet_int8_b1.bmodel 用于BM1684的INT8 BModel，batch_size=1
│   └── resnet_int8_b4.bmodel 用于BM1684的INT8 BModel，batch_size=4
├── BM1684X
│   ├── resnet_fp32_b1.bmodel 用于BM1684X的FP32 BModel，batch_size=1
│   ├── resnet_fp32_b4.bmodel 用于BM1684X的FP32 BModel，batch_size=4
│   ├── resnet_int8_b1.bmodel 用于BM1684X的INT8 BModel，batch_size=1
│   └── resnet_int8_b4.bmodel 用于BM1684X的INT8 BModel，batch_size=4
└── torch
    ├── resnet50-11ad3fa6.pth  原始模型
    ├── resnet50-11ad3fa6_traced_b1.pt trace后的模型,batch_size为1
    └── resnet50-11ad3fa6_traced_b4.pt trace后的模型,batch_size为4
```

测试数据集imagenet_val_1k包括：
```
img: 测试图片，共1000张
label.txt：标签文件
```

模型信息：

| 原始模型 | resnet50-11ad3fa6.pth | 
| ------- | ----------------------   |
| 概述     | resnet50图像分类 | 
| 骨干网络 | resnet50                   | 
| 训练集   | IMAGENET1K_V2                | 
| 输入数据 | [batch_size, 3, 224, 224], FP32，NCHW |
| 输出数据 | [batch_size, 1000], FP32 |
| 前处理   | resize,BGR2RGB,减均值,除方差,HWC->CHW |
| 后处理   | softmax                 |


## 4. 模型编译

trace后的pytorch模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。

模型编译前需要安装TPU-NNTC(>=3.1.0)，具体可参考[tpu-nntc环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-nntc环境搭建)。

### 4.1 生成FP32 BModel

pytorch模型编译为FP32 BModel，具体方法可参考[BMNETP 使用](https://doc.sophgo.com/docs/3.0.0/docs_latest_release/nntc/html/usage/bmnetp.html)。

本例程在`scripts`目录下提供了编译FP32 BModel的脚本。请注意修改`gen_fp32bmodel.sh`中的JIT模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（支持BM1684和BM1684X），如：

```bash
./scripts/gen_fp32bmodel.sh BM1684X
```

执行上述命令会在`data/models/BM1684X/`下生成`resnet_fp32_b1.bmodel、resnet_fp32_b4.bmodel`文件，即转换好的FP32 BModel。


### 4.2 生成INT8 BModel

不量化模型可跳过本节。

pytorch模型的量化方法可参考[Quantization-Tools User Guide](https://doc.sophgo.com/docs/3.0.0/docs_latest_release/calibration-tools/html/index.html)

本例程在`scripts`目录下提供了量化INT8 BModel的脚本。请注意修改`gen_int8model.sh`中的JIT模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台（支持BM1684和BM1684X），如：


```shell
./scripts/gen_int8bmodel.sh BM1684X
```

上述脚本会在`data/models/BM1684X`下生成`resnet_int8_b4.bmodel、resnet_int8_b1.bmodel`文件，即转换好的INT8 BModel。


## 5. 例程测试
* [C++例程](cpp/README.md)
* [Python例程](python/README.md)



