# CLIP <!-- omit in toc -->

## 目录 <!-- omit in toc -->
- [1. 简介](#1-简介)
- [2. 特性](#2-特性)
- [3. 准备数据与模型](#3-准备数据与模型)
- [4. 模型编译](#4-模型编译)
- [5. 例程测试](#5-例程测试)
- [6. 性能测试](#6-性能测试)
  - [6.1 bmrt\_test](#61-bmrt_test)
  - [6.2 程序运行性能](#62-程序运行性能)

## 1. 简介

CLIP（Contrastive Language-Image Pre-Training）是一个在多种（图像，文本）配对上训练的神经网络。它可以用自然语言进行指导，以预测给定图像最相关的文本片段，而无需直接针对该任务进行优化，这与GPT-2和3的零样本（zero-shot）能力类似。本例程对[CLIP官方开源仓库](https://github.com/openai/CLIP)中的算法进行移植，使之能在SOPHON BM1684X,BM1688,CV186X上进行推理。

[[Blog]](https://openai.com/blog/clip/) [[Paper]](https://arxiv.org/abs/2103.00020)

## 2. 特性

* 支持BM1688/CV186X(SoC)、BM1684X(x86 PCIe、SoC)
* 支持FP16(BM1684X/BM1688/CV186X)模型编译和推理
* 支持Python例程
* 支持单batch和多batch模型推理
* 支持图片测试

## 3. 准备数据与模型

Pytorch模型在编译前要导出成onnx模型，具体可参考[CLIP模型导出](./docs/Clip_Export_Guide.md)。
​
本例程在`scripts`目录下提供了相关模型和数据集的下载脚本`download.sh`，您也可以自己准备模型和数据集，并参考[4. 模型编译](#4-模型编译)进行模型转换。

```bash
# 安装7z和zip，若已安装请跳过，非ubuntu系统视情况使用yum或其他方式安装
sudo apt install unzip
sudo apt install p7zip p7zip-full
chmod -R +x scripts/
./scripts/download.sh
```

下载的模型包括：
```
.models
├── BM1684X
│   ├── clip_image_vitb32_bm1684x_f16_1b.bmodel         # encode_image部分fp16 bmodel
│   └── clip_text_vitb32_bm1684x_f16_1b.bmodel          # encode_text部分fp16 bmodel
├── BM1688
│   ├── clip_image_vitb32_bm1688_f16_1b_2core.bmodel    # encode_image部分fp16 bmodel，num_core=2
│   ├── clip_image_vitb32_bm1688_f16_1b.bmodel          # encode_image部分fp16 bmodel
│   ├── clip_text_vitb32_bm1688_f16_1b_2core.bmodel     # encode_text部分fp16 bmodel，num_core=2
│   └── clip_text_vitb32_bm1688_f16_1b.bmodel           # encode_text部分fp16 bmodel
├── CV186X
│   ├── clip_image_vitb32_cv186x_f16_1b.bmodel          # encode_image部分fp16 bmodel，num_core=2
│   └── clip_text_vitb32_cv186x_f16_1b.bmodel           # encode_text部分fp16 bmodel，num_core=2
├── onnx
│   ├── clip_image_vitb32.onnx                          # encode_image部分onnx模型
│   └── clip_text_vitb32.onnx                           # encode_text部分onnx模型
└── text_projection_512_512.npy                         # 导出encode_text onnx模型时保存的text_projection数据，在bmodel推理时使用
```


## 4. 模型编译

导出的模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。

模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#2-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index/material/all/all.html)相应版本的SDK中获取)。

- 生成FP16 BModel

本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x #bm1688/cv186x
```

执行上述命令会在`models/BM1684X/`下生成`CLIP_fp16_1b.bmodel`文件，即转换好的FP16 BModel。

## 5. 例程测试

- [Python例程](./python/README.md)

## 6. 性能测试


### 6.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684X/clip_image_vitb32_bm1684x_f16_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

| 测试clip_image_vitb32模型                           | calculate time(ms) |
| --------------------------------------------------- | ------------------ |
| BM1684X/clip_image_vitb32_bm1684x_f16_1b.bmodel     | 19.16              |
| BM1688/clip_image_vitb32_bm1688_f16_1b.bmodel       | 13.67              |
| BM1688/clip_image_vitb32_bm1688_f16_1b_2core.bmodel | 18.82              |
| CV186X/clip_image_vitb32_cv186x_f16_1b.bmodel       | 25.79              |

| 测试clip_text_vitb32模型                           | calculate time(ms) |
| -------------------------------------------------- | ------------------ |
| BM1684X/clip_text_vitb32_bm1684x_f16_1b.bmodel     | 4.92               |
| BM1688/clip_text_vitb32_bm1688_f16_1b.bmodel       | 13.71              |
| BM1688/clip_text_vitb32_bm1688_f16_1b_2core.bmodel | 14.08              |
| CV186X/clip_text_vitb32_cv186x_f16_1b.bmodel       | 17.61              |


> **测试说明**：
> 1. 性能测试结果具有一定的波动性；
> 2. `calculate time`已折算为平均每张图片的推理时间；
> 3. SoC和PCIe的测试结果基本一致。

### 6.2 程序运行性能
测试图片`/datasets/CLIP.png`，测试模型为clip_image_vitb32_\<target\>_f16_1b.bmodel，clip_text_vitb32_\<target\>_f16_1b.bmodel

测试结果如下，测试结果有一定波动性，取稳定后的性能数据（时间单位为ms）：

| 测试平台 | 测试程序            | Preprocess Time | Image Encoding Time | Text Encoding Time |
| -------- | ------------------- | --------------- | ------------------- | ------------------ |
| SE7-32   | zeroshot_predict.py | 12.17           | 9.63                | 18.90              |
| SE9-16   | zeroshot_predict.py | 16.92           | 25.04               | 49.61              |
| SE9-8    | zeroshot_predict.py | 17.09           | 30.59               | 59.56              |

> **测试说明**：
> 1. 性能测试结果具有一定的波动性，实测结果与该表结果有误差属正常现象，建议取稳定后的性能数据、并多次测试取平均值。
> 2. 初次启动程序，程序解码、推理时间较长，再次运行程序时间正常，为正常现象，原因是文件还没有缓存到cache中。
> 3. SE7-32的主控处理器为8核CA53@2.3GHz，SE9-16为8核CA53@1.6GHz，SE9-8为6核CA53@1.6GHz，PCIe上的性能由于处理器的不同可能存在较大差异。
