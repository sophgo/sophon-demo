# BLIP <!-- omit in toc -->

## 目录 <!-- omit in toc -->
- [1. 简介](#1-简介)
- [2. 特性](#2-特性)
- [3. 运行环境准备](#3-运行环境准备)
- [4. 准备数据与模型](#4-准备数据与模型)
- [5. 模型编译](#5-模型编译)
- [6. 例程测试](#6-例程测试)
- [7. 性能测试](#7-性能测试)
  - [7.1 bmrt\_test](#71-bmrt_test)
  - [7.2 程序运行性能](#72-程序运行性能)

## 1. 简介

BLIP (Bootstrapping Language-Image Pre-training) 是一种用于多模态学习的模型，旨在处理图像和语言的结合任务，如图像字幕生成和视觉问答。BLIP 的核心思想是通过结合语言和图像的特征表示来进行预训练，从而提高模型在多模态任务中的表现。本例程对[BLIP官方开源仓库](https://github.com/salesforce/BLIP)中的算法进行移植，使之能在SOPHON BM1684,BM1684X,BM1688上进行推理。

[[Blog]](https://openai.com/blog/blip/) [[Paper]](https://arxiv.org/abs/2103.00020)

## 2. 特性

* 支持BM1688(SoC)、BM1684/BM1684X(x86 PCIe、SoC)
* 支持Python例程
* 支持图片测试

## 3. 运行环境准备
在PCIe上无需修改内存，以下为soc模式相关：

对于1684/1684X系列设备（如SE5/SE7/SM5/SM7），确保使用V24.04.01刷机包; 
对于1688设备（SE9），确保使用v1.6刷机包；
以上刷机包都可从算能官网发布的SDK中获取；

确保SDK版本后，在1684x SoC环境上，参考如下命令修改设备内存。

```bash
cd /data/
mkdir memedit && cd memedit
wget -nd https://sophon-file.sophon.cn/sophon-prod-s3/drive/23/09/11/13/DeviceMemoryModificationKit.tgz
tar xvf DeviceMemoryModificationKit.tgz
cd DeviceMemoryModificationKit
tar xvf memory_edit_{vx.x}.tar.xz #vx.x是版本号
cd memory_edit
./memory_edit.sh -p #这个命令会打印当前的内存布局信息
# BM1684/BM1684X设备执行下列语句
./memory_edit.sh -c -npu 7615 -vpu 3072 -vpp 3072 #npu也可以访问vpu和vpp的内存
sudo cp /data/memedit/DeviceMemoryModificationKit/memory_edit/emmcboot.itb /boot/emmcboot.itb && sync
# BM1688设备执行下列语句
./memory_edit.sh -c -npu 6328 -vpu 0 -vpp 512
sudo cp /data/memedit/DeviceMemoryModificationKit/memory_edit/boot.itb /boot/boot.itb && sync
# 重启生效
sudo reboot
```

> **注意：**
> 1. tpu总内存为npu/vpu/vpp三者之和。
> 2. 更多教程请参考[SoC内存修改工具](https://doc.sophgo.com/sdk-docs/v24.04.01/docs_latest_release/docs/SophonSDK_doc/zh/html/appendix/2_mem_edit_tools.html#)

## 4. 准备数据与模型

Pytorch模型在编译前要导出成onnx模型，具体可参考[BLIP模型导出](./docs/Blip_Export_Guide.md)。
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
models
├── bert-base-uncased                            # tokenizer, bert分词器
├── BM1684
│   ├── blip_cap_bm1684_f32_1b.bmodel            # 图像字幕 1684 fp32 bmodel
│   ├── blip_itm_bm1684_f32_1b.bmodel            # 图文匹配 1684 fp32 bmodel
│   ├── blip_vqa_tdec_bm1684_f32_1b.bmodel       # 图文问答 解码器 1684 fp32 bmodel
│   ├── blip_vqa_tenc_bm1684_f32_1b.bmodel       # 图文问答 图文编码器 1684 fp32 bmodel
│   └── blip_vqa_venc_bm1684_f32_1b.bmodel       # 图文问答 图像编码器 1684 fp32 bmodel
├── BM1684X
│   ├── blip_cap_bm1684x_f32_1b.bmodel           # 图像字幕 1684X fp32 bmodel
│   ├── blip_itm_bm1684x_f32_1b.bmodel           # 图文匹配 1684X fp32 bmodel
│   ├── blip_vqa_tdec_bm1684x_f32_1b.bmodel      # 图文问答 解码器 1684X fp32 bmodel
│   ├── blip_vqa_tenc_bm1684x_f32_1b.bmodel      # 图文问答 图文编码器 1684X fp32 bmodel
│   └── blip_vqa_venc_bm1684x_f32_1b.bmodel      # 图文问答 图像编码器 1684X fp32 bmodel
└── BM1688
    ├── blip_cap_bm1688_f32_1b.bmodel            # 图像字幕 1688 fp32 bmodel
    ├── blip_itm_bm1688_f32_1b.bmodel            # 图文匹配 1688 fp32 bmodel
    ├── blip_vqa_tdec_bm1688_f32_1b.bmodel       # 图文问答 解码器 1688 fp32 bmodel
    ├── blip_vqa_tenc_bm1688_f32_1b.bmodel       # 图文问答 图文编码器 1688 fp32 bmodel
    └── blip_vqa_venc_bm1688_f32_1b.bmodel       # 图文问答 图像编码器 1688 fp32 bmodel
```


## 5. 模型编译

导出的模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。

模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#2-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index/material/all/all.html)相应版本的SDK中获取)。

- 生成FP32 BModel

本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684/BM1684X/BM1688**），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684x #bm1684/bm1688
```

执行上述命令会在`models/BM1684X/`下生成`blip_*_bm1684x_f32_1b.bmodel`文件，即转换好的FP32 BModel。

## 6. 例程测试

- [Python例程](./python/README.md)
- [WebUI例程](./web_ui/README.md)

## 7. 性能测试


### 7.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684X/blip_itm_bm1684_f32_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

| 测试平台 | 测试blip_cap模型                           | calculate time(ms) |
|---------|--------------------------------------------| ------------------ |
| SE7-32  | BM1684X/blip_cap_bm1684x_f32_1b.bmodel     | 332                |
| SE9-16  | BM1688/blip_cap_bm1688_f32_1b.bmodel       | 987               |


| 测试平台 | 测试blip_itm模型                           | calculate time(ms) |
| --------|------------------------------------------- | ------------------ |
| SE5-16  | BM1684/blip_itm_bm1684_f32_1b.bmodel       | 513                |
| SE7-32  | BM1684X/blip_itm_bm1684x_f32_1b.bmodel     | 201                |
| SE9-16  | BM1688/blip_itm_bm1688_f32_1b.bmodel       | 798                |


| 测试平台 | 测试blip_vqa模型                                 | calculate time(ms) |
| --------|------------------------------------------------ | ------------------ |
| SE5-16  | BM1684/blip_vqa_venc_bm1684_f32_1b.bmodel       | 742                |
| SE5-16  | BM1684/blip_vqa_tenc_bm1684_f32_1b.bmodel       | 110                |
| SE5-16  | BM1684/blip_vqa_tdec_bm1684_f32_1b.bmodel       | 86                 |
| SE7-32  | BM1684X/blip_vqa_venc_bm1684x_f32_1b.bmodel     | 352                |
| SE7-32  | BM1684X/blip_vqa_tenc_bm1684x_f32_1b.bmodel     | 11                 |
| SE7-32  | BM1684X/blip_vqa_tdec_bm1684x_f32_1b.bmodel     | 46                 |
| SE9-16  | BM1688/blip_vqa_venc_bm1688_f32_1b.bmodel       | 1090               |
| SE9-16  | BM1688/blip_vqa_tenc_bm1688_f32_1b.bmodel       | 202                |
| SE9-16  | BM1688/blip_vqa_tdec_bm1688_f32_1b.bmodel       | 101                |



> **测试说明**：
> 1. 性能测试结果具有一定的波动性；
> 2. `calculate time`已折算为平均每张图片的推理时间；

### 7.2 程序运行性能
测试图片`/datasets/test/demo.png`，测试模型均为对应的fp32模型

测试结果如下，测试结果有一定波动性，取稳定后的性能数据（时间单位为ms）：

| 测试平台 | 测试程序      | Preprocess Time | Predict Time |
| -------- | ----------- | --------------- | ------------- |
| SE5-16   | blip_itm.py | 48.85           | 519.43        |
| SE5-16   | blip_vqa.py | 16.21           | 202.65        |
| SE7-32   | blip_cap.py | 39.15           | 338.35        |
| SE7-32   | blip_itm.py | 26.05           | 204.82        |
| SE7-32   | blip_vqa.py | 19.32           |  60.26        |
| SE9-16   | blip_cap.py | 76.05           |1000.13        |
| SE9-16   | blip_itm.py | 80.74           | 807.94        |
| SE9-16   | blip_vqa.py | 19.01           | 309.29        |

> **测试说明**：
> 1. 性能测试结果具有一定的波动性，实测结果与该表结果有误差属正常现象，建议取稳定后的性能数据、并多次测试取平均值。
> 2. 初次启动程序，程序解码、推理时间较长，再次运行程序时间正常，为正常现象，原因是文件还没有缓存到cache中。
> 3. SE7-32的主控处理器为8核CA53@2.3GHz，SE9-16为8核CA53@1.6GHz，SE9-8为6核CA53@1.6GHz，PCIe上的性能由于处理器的不同可能存在较大差异。
