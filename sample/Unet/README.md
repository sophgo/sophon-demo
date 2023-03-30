# Unet

## 目录

- [Unet](#unet)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 数据集](#2-数据集)
  - [3. 准备模型与数据](#3-准备模型与数据)
  - [4. 模型转换](#4-模型转换)
    - [4.1 生成fp32 bmodel](#41-生成fp32-bmodel)
  - [5. 例程测试](#5-例程测试)
  - [6. 精度测试](#6-精度测试)

## 1. 简介
本例程对[Unet]的模型和算法进行移植，使之能在SOPHON BM1684上进行推理测试。


**论文:** [Unet论文](https://arxiv.org/pdf/1505.04597.pdf)
U-Net: Convolutional Networks for Biomedical Image Segmentation，是2015年于MICCAI发表的经典的语义分割网络。原论文在医学影像领域对其性能进行了测试，但它也可以应用到更加广泛的任务上。

## 2. 数据集
本例程使用Unet对Carvana汽车数据集进行分割。Carvana是一个在线二手车销售公司,由于他们拍摄的车辆照片中车辆与背景颜色相近，使用一般的工具提取汽车区域时往往会出现分割错误的问题。Carvana数据集由Kaggle提供，其中train.zip和train_masks.zip包括5088张汽车图像对应的标注。
Carvana数据集下载地址:

图像：https://www.kaggle.com/competitions/carvana-image-masking-challenge/data?select=train.zip

标注：https://www.kaggle.com/competitions/carvana-image-masking-challenge/data?select=train_masks.zip

## 3. 准备模型与数据
您需要准备用于测试的数据集。

本例程在`scripts`目录下提供了相关模型、测试图像和测试视频的下载脚本`download.sh`，您也可以自己准备模型和数据集，并参考[4. 模型转换](#4-模型转换)进行模型转换。
```bash
chmod -R +x scripts/
./scripts/download.sh
```
执行后，模型保存至`./models`, 数据集保存至`./datasets`
```
下载的模型包括：
./models/torch/unet_carvana_scale0.5_epoch2.pth: 原始模型
./models/torch/unet.pt: 使用pytorch工具得到的jit模型
./models/BM1684/unet_fp32_1b.bmodel: 用于BM1684的FP32 bmodel, batch_size=1
./models/BM1684X/unet_fp32_1b.bmocel: 用于BM1684X的FP32 bmodel, batch_size=1
```

## 4. 模型转换
模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。

模型编译前需要安装TPU-NNTC(>=3.1.0)，具体可参考[tpu-nntc环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-nntc环境搭建)。

### 4.1 生成fp32 bmodel
模型编译为FP32 BModel，具体方法可参考[BMNETP 使用](https://doc.sophgo.com/docs/3.0.0/docs_latest_release/nntc/html/usage/bmnetp.html)。

本例程在`scripts`目录下提供了编译FP32 BModel的脚本。请注意修改`gen_fp32bmodel.sh`中的模型路径、生成模型目录和输入大小shapes等参数.

```bash
./scripts/gen_fp32bmodel.sh
```
执行上述命令会在`./models/BM1684`及`./models/BM1684X`下生成`unet_fp32_b1`文件，即转换好的FP32 BModel。

## 5. 例程测试
* [C++例程](./cpp/README.md)
* [Python例程](./python/README.md)

## 6. 精度测试
首先，参考[C++例程](cpp/README.md)或[Python例程](python/README.md)推理要测试的数据集，保存预测结果，注意修改修改相关参数(out_threshold=0.5、n_classes=2)。 

然后，使用`tools`目录下的`eval_carvana.py`脚本，将保存的图像文件与标签文件进行对比，计算出分割任务的评价指标，命令如下：
```bash
# 请根据实际情况修改图像文件路径
python3 eval_carvana.py --pred_path ../cpp/unet_bmcv/results/images --label_path ../datasets/label
```