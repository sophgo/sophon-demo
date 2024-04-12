[简体中文](./README.md) | [English](./README_EN.md)
# 人脸识别SCRFD

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
SCRFD(Sample and Computation Redistribution for Efficient Face Detection)是一种基于FCOS 的人脸检测算法，该算法在2021年5月推出。它被设计为一个高效和高精度的人脸检测器，其速度和准确性相较于其他现有算法都有显著提高。

**论文地址** (https://arxiv.org/pdf/2105.04714.pdf)

**源码地址** (https://github.com/deepinsight/insightface/tree/master/detection/scrfd)

## 2. 特性
* 支持BM1688/CV186X(SoC)、BM1684X(x86 PCIe、SoC)、BM1684(x86 PCIe、SoC)
* 支持FP32、FP16(BM1684X/BM1688/CV186X)、INT8模型编译和推理
* 支持基于BMCV预处理的C++推理
* 支持基于OpenCV和BMCV预处理的Python推理
* 支持单batch模型推理
* 支持图片和视频测试

## 3. 准备模型与数据

建议使用TPU-MLIR编译BModel，Pytorch模型在编译前要导出成onnx模型，如果您使用的tpu-mlir版本>=v1.3.0（即官网v23.07.01），可以直接使用torchscript模型。具体可参考[SCRFD模型导出方法](./docs/scrfd_Export_Guide.md)。

​同时，您需要准备用于测试的数据集，如果量化模型，还要准备用于量化的数据集。
本例程提供了一种性能和精度上较高的模型， `scrfd_10g_kps.onnx` 。您都可以使用MLIR工具链转出为对应的bmodel模型。

如果您想使用其他模型，您可以访问 [源码地址](https://github.com/deepinsight/insightface/tree/master/detection/scrfd) 进行下载，并参考 [SCRFD模型导出方法](./docs/scrfd_Export_Guide.md) 进行导出。

同时，您需要准备用于测试的数据集，如果量化模型，还要准备用于量化的数据集。

本例程在`scripts`目录下提供了相关模型和数据集的下载脚本`download.sh`，您也可以自己准备模型和数据集，并参考[4. 模型编译](#4-模型编译)进行模型转换。


```bash
# 安装unzip，若已安装请跳过，非ubuntu系统视情况使用yum或其他方式安装
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```

下载的模型包括
```
./models
.
├── BM1684                                   # 使用TPU-MLIR编译，用于BM1684的 BModel
│   ├── scrfd_10g_kps_fp32_1b.bmodel
│   ├── scrfd_10g_kps_int8_1b.bmodel
│   ├── scrfd_10g_kps_int8_4b.bmodel
├── BM1684X                                  # 使用TPU-MLIR编译，用于BM1684X的 BModel
│   ├── scrfd_10g_kps_fp16_1b.bmodel
│   ├── scrfd_10g_kps_fp32_1b.bmodel
│   ├── scrfd_10g_kps_int8_1b.bmodel
│   ├── scrfd_10g_kps_int8_4b.bmodel
├── BM1688                                   # 使用TPU-MLIR编译，用于BM1688的 BModel
│   ├── scrfd_10g_kps_fp16_1b_2core.bmodel
│   ├── scrfd_10g_kps_fp16_1b.bmodel
│   ├── scrfd_10g_kps_fp32_1b_2core.bmodel
│   ├── scrfd_10g_kps_fp32_1b.bmodel
│   ├── scrfd_10g_kps_int8_1b_2core.bmodel
│   ├── scrfd_10g_kps_int8_1b.bmodel
│   ├── scrfd_10g_kps_int8_4b_2core.bmodel
│   ├── scrfd_10g_kps_int8_4b.bmodel
├── CV186X                                   # 使用TPU-MLIR编译，用于CV186X的 BModel
│   ├── scrfd_10g_kps_fp16_1b.bmodel
│   ├── scrfd_10g_kps_fp32_1b.bmodel
│   ├── scrfd_10g_kps_int8_1b.bmodel
│   ├── scrfd_10g_kps_int8_4b.bmodel
└── onnx                                     # 导出的onnx模型
    ├── scrfd_10g_kps_1b.onnx
    ├── scrfd_10g_kps_4b.onnx
```

下载的数据包括：
```
./datasets
├── face_det.mp4                     # 测试视频
├── test                             # 测试图片
│   ├── men.jpg
│   └── selfie.jpg
└── WIDER_val                        # 精度评估数据集
    └── images
```


## 4. 模型编译
模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index/material/31/all.html)相应版本的SDK中获取)。

- 生成FP32 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684 #bm1684x/bm1688/cv186x
```

​执行上述命令会在`models/BM1684`等文件夹下生成`scrfd_10g_kps_fp32_1b.bmodel`文件，即转换好的FP32 BModel。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x #bm1688/cv186x
```

​执行上述命令会在`models/BM1684X/`等文件夹下生成`scrfd_10g_kps_fp16_1b.bmodel`文件，即转换好的FP16 BModel。

- 生成INT8 BModel

​本例程在`scripts`目录下提供了量化INT8 BModel的脚本，请注意修改`gen_int8bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台（**支持BM1684X/BM1688/CV186X**），如：

```shell
./scripts/gen_int8bmodel_mlir.sh bm1684 #bm1684x/bm1688/cv186x
```

​上述脚本会在`models/BM1684`等文件夹下生成`scrfd_10g_kps_int8_1b.bmodel`等文件，即转换好的INT8 BModel。


## 5. 例程测试
- [C++例程](./cpp/README.md)
- [Python例程](./python/README.md)


## 6. 精度测试
### 6.1 测试方法
首先，参考[C++例程](cpp/README.md#32-测试图片)或[Python例程](python/README.md#22-测试图片)推理要测试的数据集，生成预测的txt文件夹，注意修改数据集(datasets/WIDER_val)和相关参数(conf_thresh=0.02、nms_thresh=0.45以及--eval=True)。  
然后，使用`tools`目录下的`evaluation.py`脚本，将测试生成的txt文件夹与测试集标签ground_truth文件夹进行对比，计算出人脸检测的评价指标，命令如下：

```bash
cd tools
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python3 setup.py build_ext --inplace
python3 evaluation.py --pred ./prediction_dir --gt ground_truth
```
具体测试方法，请参考[精度测试](./tools/README.md)

### 6.2 测试结果
在`WIDER FACE`数据集上，官方SCRFD_10G_KPS模型的精度测试结果是：`Easy: 0.9540, Medium: 0.9401, Hard: 0.8280`，本例程的精度测试结果如下表所示：

|    测试平台    |     测试程序    |             测试模型         | Easy    |Medium  | Hard   | 
| ------------ |------------------- | ---------------------------------------- | -------- | -------  |----------|
| SE5-16       | scrfd_opencv.py    | scrfd_10g_kps_fp32_1b.bmodel             |    0.940 |    0.924 |    0.800 |
| SE5-16       | scrfd_opencv.py    | scrfd_10g_kps_int8_1b.bmodel             |    0.913 |    0.904 |    0.783 |
| SE5-16       | scrfd_opencv.py    | scrfd_10g_kps_int8_4b.bmodel             |    0.929 |    0.916 |    0.796 |
| SE5-16       | scrfd_bmcv.py      | scrfd_10g_kps_fp32_1b.bmodel             |    0.939 |    0.921 |    0.784 |
| SE5-16       | scrfd_bmcv.py      | scrfd_10g_kps_int8_1b.bmodel             |    0.912 |    0.900 |    0.767 |
| SE5-16       | scrfd_bmcv.py      | scrfd_10g_kps_int8_4b.bmodel             |    0.926 |    0.912 |    0.778 |
| SE5-16       | scrfd_bmcv.soc     | scrfd_10g_kps_fp32_1b.bmodel             |    0.936 |    0.917 |    0.764 |
| SE5-16       | scrfd_bmcv.soc     | scrfd_10g_kps_int8_1b.bmodel             |    0.835 |    0.825 |    0.659 |
| SE5-16       | scrfd_bmcv.soc     | scrfd_10g_kps_int8_4b.bmodel             |    0.864 |    0.847 |    0.677 |
| SE5-16       | scrfd_sail.soc     | scrfd_10g_kps_fp32_1b.bmodel             |    0.936 |    0.917 |    0.764 |
| SE5-16       | scrfd_sail.soc     | scrfd_10g_kps_int8_1b.bmodel             |    0.835 |    0.825 |    0.659 |
| SE5-16       | scrfd_sail.soc     | scrfd_10g_kps_int8_4b.bmodel             |    0.864 |    0.847 |    0.677 |
| SE7-32       | scrfd_opencv.py    | scrfd_10g_kps_fp32_1b.bmodel             |    0.940 |    0.924 |    0.800 |
| SE7-32       | scrfd_opencv.py    | scrfd_10g_kps_fp16_1b.bmodel             |    0.940 |    0.924 |    0.800 |
| SE7-32       | scrfd_opencv.py    | scrfd_10g_kps_int8_1b.bmodel             |    0.939 |    0.923 |    0.796 |
| SE7-32       | scrfd_opencv.py    | scrfd_10g_kps_int8_4b.bmodel             |    0.939 |    0.923 |    0.799 |
| SE7-32       | scrfd_bmcv.py      | scrfd_10g_kps_fp32_1b.bmodel             |    0.939 |    0.921 |    0.786 |
| SE7-32       | scrfd_bmcv.py      | scrfd_10g_kps_fp16_1b.bmodel             |    0.939 |    0.921 |    0.786 |
| SE7-32       | scrfd_bmcv.py      | scrfd_10g_kps_int8_1b.bmodel             |    0.938 |    0.919 |    0.783 |
| SE7-32       | scrfd_bmcv.py      | scrfd_10g_kps_int8_4b.bmodel             |    0.937 |    0.919 |    0.783 |
| SE7-32       | scrfd_bmcv.soc     | scrfd_10g_kps_fp32_1b.bmodel             |    0.937 |    0.917 |    0.772 |
| SE7-32       | scrfd_bmcv.soc     | scrfd_10g_kps_fp16_1b.bmodel             |    0.937 |    0.917 |    0.772 |
| SE7-32       | scrfd_bmcv.soc     | scrfd_10g_kps_int8_1b.bmodel             |    0.885 |    0.863 |    0.689 |
| SE7-32       | scrfd_bmcv.soc     | scrfd_10g_kps_int8_4b.bmodel             |    0.887 |    0.865 |    0.691 |
| SE7-32       | scrfd_sail.soc     | scrfd_10g_kps_fp32_1b.bmodel             |    0.937 |    0.917 |    0.772 |
| SE7-32       | scrfd_sail.soc     | scrfd_10g_kps_fp16_1b.bmodel             |    0.937 |    0.917 |    0.772 |
| SE7-32       | scrfd_sail.soc     | scrfd_10g_kps_int8_1b.bmodel             |    0.885 |    0.863 |    0.689 |
| SE7-32       | scrfd_sail.soc     | scrfd_10g_kps_int8_4b.bmodel             |    0.887 |    0.864 |    0.690 |
| SE9-16       | scrfd_opencv.py    | scrfd_10g_kps_fp32_1b.bmodel             |    0.940 |    0.924 |    0.800 |
| SE9-16       | scrfd_opencv.py    | scrfd_10g_kps_fp16_1b.bmodel             |    0.940 |    0.924 |    0.800 |
| SE9-16       | scrfd_opencv.py    | scrfd_10g_kps_int8_1b.bmodel             |    0.938 |    0.923 |    0.798 |
| SE9-16       | scrfd_opencv.py    | scrfd_10g_kps_int8_4b.bmodel             |    0.939 |    0.923 |    0.798 |
| SE9-16       | scrfd_bmcv.py      | scrfd_10g_kps_fp32_1b.bmodel             |    0.938 |    0.919 |    0.780 |
| SE9-16       | scrfd_bmcv.py      | scrfd_10g_kps_fp16_1b.bmodel             |    0.938 |    0.919 |    0.780 |
| SE9-16       | scrfd_bmcv.py      | scrfd_10g_kps_int8_1b.bmodel             |    0.936 |    0.917 |    0.776 |
| SE9-16       | scrfd_bmcv.py      | scrfd_10g_kps_int8_4b.bmodel             |    0.936 |    0.917 |    0.776 |
| SE9-16       | scrfd_bmcv.soc     | scrfd_10g_kps_fp32_1b.bmodel             |    0.936 |    0.916 |    0.766 |
| SE9-16       | scrfd_bmcv.soc     | scrfd_10g_kps_fp16_1b.bmodel             |    0.936 |    0.916 |    0.766 |
| SE9-16       | scrfd_bmcv.soc     | scrfd_10g_kps_int8_1b.bmodel             |    0.886 |    0.864 |    0.687 |
| SE9-16       | scrfd_bmcv.soc     | scrfd_10g_kps_int8_4b.bmodel             |    0.886 |    0.864 |    0.687 |
| SE9-16       | scrfd_sail.soc     | scrfd_10g_kps_fp32_1b.bmodel             |    0.930 |    0.912 |    0.764 |
| SE9-16       | scrfd_sail.soc     | scrfd_10g_kps_fp16_1b.bmodel             |    0.935 |    0.915 |    0.765 |
| SE9-16       | scrfd_sail.soc     | scrfd_10g_kps_int8_1b.bmodel             |    0.885 |    0.863 |    0.687 |
| SE9-16       | scrfd_sail.soc     | scrfd_10g_kps_int8_4b.bmodel             |    0.885 |    0.863 |    0.686 |
| SE9-16       | scrfd_opencv.py    | scrfd_10g_kps_fp32_1b_2core.bmodel       |    0.940 |    0.924 |    0.800 |
| SE9-16       | scrfd_opencv.py    | scrfd_10g_kps_fp16_1b_2core.bmodel       |    0.940 |    0.924 |    0.800 |
| SE9-16       | scrfd_opencv.py    | scrfd_10g_kps_int8_1b_2core.bmodel       |    0.938 |    0.923 |    0.798 |
| SE9-16       | scrfd_opencv.py    | scrfd_10g_kps_int8_4b_2core.bmodel       |    0.939 |    0.923 |    0.798 |
| SE9-16       | scrfd_bmcv.py      | scrfd_10g_kps_fp32_1b_2core.bmodel       |    0.938 |    0.919 |    0.780 |
| SE9-16       | scrfd_bmcv.py      | scrfd_10g_kps_fp16_1b_2core.bmodel       |    0.938 |    0.919 |    0.780 |
| SE9-16       | scrfd_bmcv.py      | scrfd_10g_kps_int8_1b_2core.bmodel       |    0.936 |    0.917 |    0.776 |
| SE9-16       | scrfd_bmcv.py      | scrfd_10g_kps_int8_4b_2core.bmodel       |    0.936 |    0.917 |    0.776 |
| SE9-16       | scrfd_bmcv.soc     | scrfd_10g_kps_fp32_1b_2core.bmodel       |    0.936 |    0.916 |    0.766 |
| SE9-16       | scrfd_bmcv.soc     | scrfd_10g_kps_fp16_1b_2core.bmodel       |    0.936 |    0.916 |    0.766 |
| SE9-16       | scrfd_bmcv.soc     | scrfd_10g_kps_int8_1b_2core.bmodel       |    0.886 |    0.864 |    0.687 |
| SE9-16       | scrfd_bmcv.soc     | scrfd_10g_kps_int8_4b_2core.bmodel       |    0.886 |    0.864 |    0.687 |
| SE9-16       | scrfd_sail.soc     | scrfd_10g_kps_fp32_1b_2core.bmodel       |    0.931 |    0.913 |    0.764 |
| SE9-16       | scrfd_sail.soc     | scrfd_10g_kps_fp16_1b_2core.bmodel       |    0.931 |    0.912 |    0.764 |
| SE9-16       | scrfd_sail.soc     | scrfd_10g_kps_int8_1b_2core.bmodel       |    0.884 |    0.862 |    0.686 |
| SE9-16       | scrfd_sail.soc     | scrfd_10g_kps_int8_4b_2core.bmodel       |    0.885 |    0.863 |    0.686 |
| SE9-8        | scrfd_opencv.py    | scrfd_10g_kps_fp32_1b.bmodel             |    0.940 |    0.924 |    0.800 |
| SE9-8        | scrfd_opencv.py    | scrfd_10g_kps_fp16_1b.bmodel             |    0.940 |    0.924 |    0.800 |
| SE9-8        | scrfd_opencv.py    | scrfd_10g_kps_int8_1b.bmodel             |    0.938 |    0.923 |    0.798 |
| SE9-8        | scrfd_opencv.py    | scrfd_10g_kps_int8_4b.bmodel             |    0.939 |    0.923 |    0.799 |
| SE9-8        | scrfd_bmcv.py      | scrfd_10g_kps_fp32_1b.bmodel             |    0.938 |    0.919 |    0.778 |
| SE9-8        | scrfd_bmcv.py      | scrfd_10g_kps_fp16_1b.bmodel             |    0.938 |    0.919 |    0.778 |
| SE9-8        | scrfd_bmcv.py      | scrfd_10g_kps_int8_1b.bmodel             |    0.936 |    0.917 |    0.776 |
| SE9-8        | scrfd_bmcv.py      | scrfd_10g_kps_int8_4b.bmodel             |    0.936 |    0.917 |    0.776 |
| SE9-8        | scrfd_bmcv.soc     | scrfd_10g_kps_fp32_1b.bmodel             |    0.936 |    0.916 |    0.765 |
| SE9-8        | scrfd_bmcv.soc     | scrfd_10g_kps_fp16_1b.bmodel             |    0.936 |    0.916 |    0.765 |
| SE9-8        | scrfd_bmcv.soc     | scrfd_10g_kps_int8_1b.bmodel             |    0.885 |    0.863 |    0.687 |
| SE9-8        | scrfd_bmcv.soc     | scrfd_10g_kps_int8_4b.bmodel             |    0.886 |    0.864 |    0.688 |
| SE9-8        | scrfd_sail.soc     | scrfd_10g_kps_fp32_1b.bmodel             |    0.936 |    0.916 |    0.765 |
| SE9-8        | scrfd_sail.soc     | scrfd_10g_kps_fp16_1b.bmodel             |    0.936 |    0.916 |    0.765 |
| SE9-8        | scrfd_sail.soc     | scrfd_10g_kps_int8_1b.bmodel             |    0.885 |    0.863 |    0.687 |
| SE9-8        | scrfd_sail.soc     | scrfd_10g_kps_int8_4b.bmodel             |    0.885 |    0.863 |    0.687 |



## 7. 性能测试
### 7.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684X/scrfd_10g_kps_fp32_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

|                测试模型                    |calculate time(ms)|
| ------------------------------------------ | -----------------|
| BM1684/scrfd_10g_kps_fp32_1b.bmodel        |           20.082 |
| BM1684/scrfd_10g_kps_int8_1b.bmodel        |           16.265 |
| BM1684/scrfd_10g_kps_int8_4b.bmodel        |            5.024 |
| BM1684X/scrfd_10g_kps_fp16_1b.bmodel       |            3.791 |
| BM1684X/scrfd_10g_kps_fp32_1b.bmodel       |           34.830 |
| BM1684X/scrfd_10g_kps_int8_1b.bmodel       |            2.645 |
| BM1684X/scrfd_10g_kps_int8_4b.bmodel       |            2.537 |
| BM1688/scrfd_10g_kps_fp16_1b.bmodel        |           45.524 |
| BM1688/scrfd_10g_kps_fp16_1b_2core.bmodel  |           31.586 |
| BM1688/scrfd_10g_kps_fp32_1b.bmodel        |          323.095 |
| BM1688/scrfd_10g_kps_fp32_1b_2core.bmodel  |          190.639 |
| BM1688/scrfd_10g_kps_int8_1b.bmodel        |           13.398 |
| BM1688/scrfd_10g_kps_int8_1b_2core.bmodel  |           11.044 |
| BM1688/scrfd_10g_kps_int8_4b.bmodel        |           12.720 |
| BM1688/scrfd_10g_kps_int8_4b_2core.bmodel  |            7.188 |
| CV186X/scrfd_10g_kps_fp16_1b.bmodel        |           42.652 |
| CV186X/scrfd_10g_kps_fp32_1b.bmodel        |          317.354 |
| CV186X/scrfd_10g_kps_int8_1b.bmodel        |           13.034 |
| CV186X/scrfd_10g_kps_int8_4b.bmodel        |           12.323 |


> **测试说明**：  
>
> 1. 性能测试结果具有一定的波动性；
> 2. `calculate time`已折算为平均每张图片的推理时间；
> 3. SoC和PCIe的测试结果基本一致。

### 7.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。

在不同的测试平台上，使用不同的例程、模型测试`datasets/WIDER_val`，conf_thresh=0.5，nms_thresh=0.5，性能测试结果如下：
|    测试平台  |     测试程序      |     测试模型          |decode_time|preprocess_time|inference_time|postprocess_time|
| ----------- | --------------- | ----------------------------------- | ----- | ----- | ------ | ----- |
|   SE5-16    |  scrfd_opencv.py  |   scrfd_10g_kps_fp32_1b.bmodel    |      35.41      |      24.20      |      25.04      |      8.57       |
|   SE5-16    |  scrfd_opencv.py  |   scrfd_10g_kps_int8_1b.bmodel    |      30.14      |      24.38      |      21.21      |      8.40       |
|   SE5-16    |  scrfd_opencv.py  |   scrfd_10g_kps_int8_4b.bmodel    |      30.24      |      26.42      |      8.70       |      8.33       |
|   SE5-16    |   scrfd_bmcv.py   |   scrfd_10g_kps_fp32_1b.bmodel    |      3.57       |      3.78       |      21.97      |      8.68       |
|   SE5-16    |   scrfd_bmcv.py   |   scrfd_10g_kps_int8_1b.bmodel    |      3.56       |      3.78       |      18.13      |      8.51       |
|   SE5-16    |   scrfd_bmcv.py   |   scrfd_10g_kps_int8_4b.bmodel    |      3.38       |      3.63       |      5.97       |      8.43       |
|   SE5-16    |  scrfd_bmcv.soc   |   scrfd_10g_kps_fp32_1b.bmodel    |      4.41       |      0.97       |      20.06      |      6.42       |
|   SE5-16    |  scrfd_bmcv.soc   |   scrfd_10g_kps_int8_1b.bmodel    |      4.40       |      0.97       |      16.24      |      6.39       |
|   SE5-16    |  scrfd_bmcv.soc   |   scrfd_10g_kps_int8_4b.bmodel    |      4.20       |      0.91       |      5.03       |      6.93       |
|   SE5-16    |  scrfd_sail.soc   |   scrfd_10g_kps_fp32_1b.bmodel    |      3.26       |      3.92       |      20.37      |      24.64      |
|   SE5-16    |  scrfd_sail.soc   |   scrfd_10g_kps_int8_1b.bmodel    |      3.21       |      3.92       |      16.55      |      24.65      |
|   SE5-16    |  scrfd_sail.soc   |   scrfd_10g_kps_int8_4b.bmodel    |      3.03       |      3.67       |      5.13       |      24.95      |
|   SE7-32    |  scrfd_opencv.py  |   scrfd_10g_kps_fp32_1b.bmodel    |      29.94      |      25.92      |      40.39      |      8.57       |
|   SE7-32    |  scrfd_opencv.py  |   scrfd_10g_kps_fp16_1b.bmodel    |      30.02      |      25.67      |      9.38       |      8.67       |
|   SE7-32    |  scrfd_opencv.py  |   scrfd_10g_kps_int8_1b.bmodel    |      30.01      |      25.65      |      8.18       |      8.68       |
|   SE7-32    |  scrfd_opencv.py  |   scrfd_10g_kps_int8_4b.bmodel    |      30.18      |      27.54      |      6.70       |      8.43       |
|   SE7-32    |   scrfd_bmcv.py   |   scrfd_10g_kps_fp32_1b.bmodel    |      3.02       |      2.97       |      36.84      |      8.75       |
|   SE7-32    |   scrfd_bmcv.py   |   scrfd_10g_kps_fp16_1b.bmodel    |      3.04       |      2.96       |      5.81       |      9.06       |
|   SE7-32    |   scrfd_bmcv.py   |   scrfd_10g_kps_int8_1b.bmodel    |      3.02       |      2.97       |      4.66       |      8.85       |
|   SE7-32    |   scrfd_bmcv.py   |   scrfd_10g_kps_int8_4b.bmodel    |      2.86       |      2.80       |      3.56       |      8.46       |
|   SE7-32    |  scrfd_bmcv.soc   |   scrfd_10g_kps_fp32_1b.bmodel    |      3.88       |      0.87       |      34.84      |      6.39       |
|   SE7-32    |  scrfd_bmcv.soc   |   scrfd_10g_kps_fp16_1b.bmodel    |      3.90       |      0.87       |      3.80       |      6.37       |
|   SE7-32    |  scrfd_bmcv.soc   |   scrfd_10g_kps_int8_1b.bmodel    |      3.90       |      0.87       |      2.65       |      6.39       |
|   SE7-32    |  scrfd_bmcv.soc   |   scrfd_10g_kps_int8_4b.bmodel    |      3.71       |      0.84       |      2.53       |      6.84       |
|   SE7-32    |  scrfd_sail.soc   |   scrfd_10g_kps_fp32_1b.bmodel    |      2.69       |      3.15       |      35.15      |      24.78      |
|   SE7-32    |  scrfd_sail.soc   |   scrfd_10g_kps_fp16_1b.bmodel    |      2.70       |      3.16       |      4.14       |      25.66      |
|   SE7-32    |  scrfd_sail.soc   |   scrfd_10g_kps_int8_1b.bmodel    |      2.69       |      3.16       |      2.98       |      25.04      |
|   SE7-32    |  scrfd_sail.soc   |   scrfd_10g_kps_int8_4b.bmodel    |      2.55       |      3.10       |      2.70       |      25.02      |
|   SE9-16    |  scrfd_opencv.py  |   scrfd_10g_kps_fp32_1b.bmodel    |      53.97      |      46.81      |     181.32      |      13.20      |
|   SE9-16    |  scrfd_opencv.py  |   scrfd_10g_kps_fp16_1b.bmodel    |      54.74      |      46.58      |      43.43      |      14.68      |
|   SE9-16    |  scrfd_opencv.py  |   scrfd_10g_kps_int8_1b.bmodel    |      56.00      |      45.49      |      28.14      |      15.90      |
|   SE9-16    |  scrfd_opencv.py  |   scrfd_10g_kps_int8_4b.bmodel    |      51.04      |      48.04      |      16.96      |      16.18      |
|   SE9-16    |   scrfd_bmcv.py   |   scrfd_10g_kps_fp32_1b.bmodel    |      12.16      |      10.85      |     176.44      |      13.21      |
|   SE9-16    |   scrfd_bmcv.py   |   scrfd_10g_kps_fp16_1b.bmodel    |      17.06      |      10.94      |      37.52      |      15.35      |
|   SE9-16    |   scrfd_bmcv.py   |   scrfd_10g_kps_int8_1b.bmodel    |      17.75      |      11.15      |      22.63      |      16.77      |
|   SE9-16    |   scrfd_bmcv.py   |   scrfd_10g_kps_int8_4b.bmodel    |      12.58      |      11.73      |      12.39      |      17.71      |
|   SE9-16    |  scrfd_bmcv.soc   |   scrfd_10g_kps_fp32_1b.bmodel    |      11.99      |      3.08       |     171.05      |      9.60       |
|   SE9-16    |  scrfd_bmcv.soc   |   scrfd_10g_kps_fp16_1b.bmodel    |      20.58      |      3.51       |      32.10      |      10.17      |
|   SE9-16    |  scrfd_bmcv.soc   |   scrfd_10g_kps_int8_1b.bmodel    |      19.33      |      3.61       |      16.78      |      10.93      |
|   SE9-16    |  scrfd_bmcv.soc   |   scrfd_10g_kps_int8_4b.bmodel    |      14.92      |      3.65       |      9.49       |      12.73      |
|   SE9-16    |  scrfd_sail.soc   |   scrfd_10g_kps_fp32_1b.bmodel    |      13.56      |      10.02      |     171.81      |      39.02      |
|   SE9-16    |  scrfd_sail.soc   |   scrfd_10g_kps_fp16_1b.bmodel    |      17.07      |      10.41      |      34.34      |      42.58      |
|   SE9-16    |  scrfd_sail.soc   |   scrfd_10g_kps_int8_1b.bmodel    |      17.73      |      10.36      |      18.59      |      43.38      |
|   SE9-16    |  scrfd_sail.soc   |   scrfd_10g_kps_int8_4b.bmodel    |      8.59       |      6.86       |      6.96       |      35.79      |
|   SE9-16    |  scrfd_opencv.py  |scrfd_10g_kps_fp32_1b_2core.bmodel |      54.07      |      46.95      |     114.79      |      13.30      |
|   SE9-16    |  scrfd_opencv.py  |scrfd_10g_kps_fp16_1b_2core.bmodel |      56.41      |      45.42      |      35.99      |      15.72      |
|   SE9-16    |  scrfd_opencv.py  |scrfd_10g_kps_int8_1b_2core.bmodel |      56.86      |      47.33      |      27.13      |      15.78      |
|   SE9-16    |  scrfd_opencv.py  |scrfd_10g_kps_int8_4b_2core.bmodel |      51.58      |      48.75      |      14.39      |      16.95      |
|   SE9-16    |   scrfd_bmcv.py   |scrfd_10g_kps_fp32_1b_2core.bmodel |      13.62      |      10.92      |     109.40      |      13.14      |
|   SE9-16    |   scrfd_bmcv.py   |scrfd_10g_kps_fp16_1b_2core.bmodel |      18.71      |      11.05      |      30.31      |      16.30      |
|   SE9-16    |   scrfd_bmcv.py   |scrfd_10g_kps_int8_1b_2core.bmodel |      19.53      |      11.38      |      22.17      |      16.47      |
|   SE9-16    |   scrfd_bmcv.py   |scrfd_10g_kps_int8_4b_2core.bmodel |      12.79      |      11.88      |      9.80       |      18.39      |
|   SE9-16    |  scrfd_bmcv.soc   |scrfd_10g_kps_fp32_1b_2core.bmodel |      14.26      |      3.21       |     103.61      |      9.63       |
|   SE9-16    |  scrfd_bmcv.soc   |scrfd_10g_kps_fp16_1b_2core.bmodel |      20.94      |      3.55       |      25.26      |      10.72      |
|   SE9-16    |  scrfd_bmcv.soc   |scrfd_10g_kps_int8_1b_2core.bmodel |      20.70      |      3.55       |      16.02      |      11.29      |
|   SE9-16    |  scrfd_bmcv.soc   |scrfd_10g_kps_int8_4b_2core.bmodel |      15.83      |      3.62       |      6.71       |      13.23      |
|   SE9-16    |  scrfd_sail.soc   |scrfd_10g_kps_fp32_1b_2core.bmodel |      14.32      |      9.95       |     105.04      |      38.02      |
|   SE9-16    |  scrfd_sail.soc   |scrfd_10g_kps_fp16_1b_2core.bmodel |      19.97      |      10.43      |      27.11      |      43.41      |
|   SE9-16    |  scrfd_sail.soc   |scrfd_10g_kps_int8_1b_2core.bmodel |      20.05      |      10.37      |      17.78      |      43.43      |
|   SE9-16    |  scrfd_sail.soc   |scrfd_10g_kps_int8_4b_2core.bmodel |      10.92      |      6.85       |      4.10       |      35.62      |
|    SE9-8    |  scrfd_opencv.py  |   scrfd_10g_kps_fp32_1b.bmodel    |      50.64      |      33.73      |     324.48      |      11.60      |
|    SE9-8    |  scrfd_opencv.py  |   scrfd_10g_kps_fp16_1b.bmodel    |      68.15      |      32.86      |      49.81      |      11.71      |
|    SE9-8    |  scrfd_opencv.py  |   scrfd_10g_kps_int8_1b.bmodel    |      72.49      |      33.02      |      20.37      |      11.67      |
|    SE9-8    |  scrfd_opencv.py  |   scrfd_10g_kps_int8_4b.bmodel    |      44.86      |      36.68      |      17.74      |      11.29      |
|    SE9-8    |   scrfd_bmcv.py   |   scrfd_10g_kps_fp32_1b.bmodel    |      12.48      |      7.23       |     320.43      |      11.63      |
|    SE9-8    |   scrfd_bmcv.py   |   scrfd_10g_kps_fp16_1b.bmodel    |      9.13       |      7.24       |      45.62      |      11.75      |
|    SE9-8    |   scrfd_bmcv.py   |   scrfd_10g_kps_int8_1b.bmodel    |      11.81      |      7.27       |      15.96      |      11.90      |
|    SE9-8    |   scrfd_bmcv.py   |   scrfd_10g_kps_int8_4b.bmodel    |      12.47      |      6.93       |      13.77      |      11.41      |
|    SE9-8    |  scrfd_bmcv.soc   |   scrfd_10g_kps_fp32_1b.bmodel    |      11.96      |      2.58       |     317.28      |      9.02       |
|    SE9-8    |  scrfd_bmcv.soc   |   scrfd_10g_kps_fp16_1b.bmodel    |      11.02      |      2.58       |      42.64      |      9.04       |
|    SE9-8    |  scrfd_bmcv.soc   |   scrfd_10g_kps_int8_1b.bmodel    |      11.93      |      2.57       |      13.00      |      8.95       |
|    SE9-8    |  scrfd_bmcv.soc   |   scrfd_10g_kps_int8_4b.bmodel    |      8.88       |      2.49       |      12.33      |      9.64       |
|    SE9-8    |  scrfd_sail.soc   |   scrfd_10g_kps_fp32_1b.bmodel    |      10.24      |      7.08       |     317.86      |      36.12      |
|    SE9-8    |  scrfd_sail.soc   |   scrfd_10g_kps_fp16_1b.bmodel    |      10.47      |      7.06       |      43.17      |      36.09      |
|    SE9-8    |  scrfd_sail.soc   |   scrfd_10g_kps_int8_1b.bmodel    |      9.85       |      7.06       |      13.51      |      35.63      |
|    SE9-8    |  scrfd_sail.soc   |   scrfd_10g_kps_int8_4b.bmodel    |      8.94       |      6.86       |      12.55      |      35.29      |


> **测试说明**：  
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. SE5-16/SE7-32的主控处理器均为8核CA53@2.3GHz，SE9-16为8核CA53@1.6GHz，SE9-8为6核CA53@1.6GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异，不同的阈值对后处理时间影响较大。 

## 8. FAQ
[常见问题解答](../../docs/FAQ.md)
