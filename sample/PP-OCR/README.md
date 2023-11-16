# PP-OCR

## 目录

* [1. 简介](#1-简介)
* [2. 特性](#2-特性)
* [3. 准备模型与数据](#3-准备模型与数据)
* [4. 模型编译](#4-模型编译)
* [5. 例程测试](#5-例程测试)
* [6. 精度测试](#6-精度测试)
* [7. 性能测试](#7-性能测试)
    
## 1. 简介

PP-OCRv3，是百度飞桨团队开源的超轻量OCR系列模型，包含文本检测、文本分类、文本识别模型，是PaddleOCR工具库的重要组成之一。支持中英文数字组合识别、竖排文本识别、长文本识别，其性能及精度较之前的PP-OCR版本均有明显提升。本例程对[PaddleOCR-release-2.6](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.6)的`ch_PP-OCRv3_xx`系列模型和算法进行移植，使之能在SOPHON BM1684/BM1684X/BM1688上进行推理测试。

## 2. 特性
* 支持BM1688(SoC)/BM1684X(x86 PCIe、SoC)/BM1684(x86 PCIe、SoC、arm PCIe)
* 支持FP32、FP16(BM1688/BM1684X)模型编译和推理
* 支持基于BMCV预处理的C++推理
* 支持基于OpenCV的Python推理
* 支持单batch、多batch、组合batch模型推理
* 支持图片数据集测试

## 3. 准备模型与数据
建议使用TPU-MLIR编译BModel，Paddle模型在编译前要导出成onnx模型。具体可参考[Paddle部署模型导出](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/beginner/model_save_load_cn.html)和[Paddle2ONNX文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/advanced/model_to_onnx_cn.html)

同时，您需要准备用于测试的数据集，如果量化模型，还要准备用于量化的数据集。

本例程使用[ICDAR-2019数据集](https://aistudio.baidu.com/aistudio/datasetdetail/177210)进行测试。
PP-OCR开源仓库提供提供的更多公开数据集：https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/dataset/datasets.md

​本例程在`scripts`目录下提供了相关模型和数据的下载脚本`download.sh`，您也可以自己准备模型和数据集，并参考[4. 模型编译](#4-模型编译)进行模型转换。

```bash
# 安装unzip，若已安装请跳过，非ubuntu系统视情况使用yum或其他方式安装
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```

下载的模型包括：
```bash
├── BM1684
│   ├── ch_PP-OCRv3_cls_fp32.bmodel # 使用TPU-MLIR编译，用于BM1684的FP32 BModel，由batch_size=1和batch_size=4的模型combine得到。
│   ├── ch_PP-OCRv3_det_fp32.bmodel # 使用TPU-MLIR编译，用于BM1684的FP32 BModel，由batch_size=1和batch_size=4的模型combine得到。
│   ├── ch_PP-OCRv3_rec_fp32.bmodel # 使用TPU-MLIR编译，用于BM1684的FP32 BModel，由batch_size=1和batch_size=4的模型combine得到。
├── BM1684X
│   ├── ch_PP-OCRv3_cls_fp16.bmodel # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，由batch_size=1和batch_size=4的模型combine得到。
│   ├── ch_PP-OCRv3_cls_fp32.bmodel # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，由batch_size=1和batch_size=4的模型combine得到。
│   ├── ch_PP-OCRv3_det_fp16.bmodel # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，由batch_size=1和batch_size=4的模型combine得到。
│   ├── ch_PP-OCRv3_det_fp32.bmodel # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，由batch_size=1和batch_size=4的模型combine得到。
│   ├── ch_PP-OCRv3_rec_fp16.bmodel # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，由batch_size=1和batch_size=4的模型combine得到。
│   ├── ch_PP-OCRv3_rec_fp32.bmodel # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，由batch_size=1和batch_size=4的模型combine得到。
├── onnx # 原始模型的onnx版本
└── paddle # 原始模型的paddlepaddle版本
```

下载的数据包括：
```bash
├── cali_set_det # 检测模型量化用的数据集
├── cali_set_rec # 分类、识别模型量化用的数据集
├── fonts # 字体
├── ppocr_keys_v1.txt # 汉字集合
├── train_full_images_0 # ICDAR-2019训练集
└── train_full_images_0.json # ICDAR-2019 ground truth文件
```

## 4. 模型编译
导出的模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。建议使用TPU-MLIR编译BModel。

模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index/material/31/all.html)相应版本的SDK中获取)。

- 生成FP32 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684/BM1684X/BM1688**），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684 #bm1684x/bm1688
```

​执行上述命令会在`models/BM1684`等文件夹下生成`ch_PP-OCRv3_det_fp32.bmodel`等文件，即转换好的FP32 BModel。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688**），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x #bm1684x/bm1688
```

​执行上述命令会在`models/BM1684X/`等文件夹下生成`ch_PP-OCRv3_det_fp16.bmodel`等文件，即转换好的FP16 BModel。

- 本例程暂时不支持量化。

## 5. 推理测试
* [C++例程](cpp/README.md)
* [Python例程](python/README.md)


## 6. 精度测试
### 6.1 测试方法
当你运行了[C++推理](cpp/README.md#3-推理测试)或[Python推理](python/README.md#24-全流程推理测试)，生成了结果文件之后，你可以使用本例程的精度测试脚本`tools/eval_icdar.py`和Ground-truth文件`datasets/train_full_images_0.json`来计算结果文件的`F-score/precision/recall`，测试命令如下：

```
python3 tools/eval_icdar.py --gt_path datasets/train_full_images_0.json --result_json python/results/ppocr_system_results_b4.json
```
输出如下：
```
F-score: 0.57488, Precision: 0.80639, Recall: 0.44665
```

### 6.2 测试结果
在ICDAR-2019数据集上，精度测试结果如下：
|   测试平台    |      测试程序           |   模型精度   | F-score |
|------------- | ----------------------- | -----------  | ------ |
| BM1684 PCIe  | ppocr_system_opencv.py  | fp32         | 0.575   |
| BM1684 PCIe  | ppocr_bmcv.pcie         | fp32         | 0.573   |
| BM1684X PCIe  | ppocr_system_opencv.py | fp32         | 0.575   |
| BM1684X PCIe  | ppocr_system_opencv.py | fp16         | 0.575   |
| BM1684X PCIe  | ppocr_bmcv.pcie        | fp32         | 0.572   |
| BM1684X PCIe  | ppocr_bmcv.pcie        | fp16         | 0.572   |
| BM1688 SoC    | ppocr_system_opencv.py | fp32         | 0.575   |
| BM1688 SoC    | ppocr_system_opencv.py | fp16         | 0.574   |
| BM1688 SoC    | ppocr_bmcv.pcie        | fp32         | 0.556   |
| BM1688 SoC    | ppocr_bmcv.pcie        | fp16         | 0.555   |


> **测试说明**：  
> 1. 模型精度为fp32(fp16)，即代表检测模型和识别模型都是fp32(fp16)的精度。
> 2. SoC和PCIe的模型精度一致；

## 7. 性能测试
### 7.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684X/ch_PP-OCRv3_det_fp32.bmodel
```
测试各个模型的理论推理时间，结果如下：

|                  测试模型            |stage| calculate time(ms) |
| ----------------------------------  |  ---| ----------------- |
| BM1684/ch_PP-OCRv3_det_fp32.bmodel  |  1  | 15.4          |
|                 ^                   |  2  | 59.7          |
| BM1684/ch_PP-OCRv3_rec_fp32.bmodel  |  1  | 2.7           |
|                 ^                   |  2  | 5.4           |
|                 ^                   |  3  | 8.5           |
|                 ^                   |  4  | 17.1          |
| BM1684/ch_PP-OCRv3_cls_fp32.bmodel  |  1  | 0.3           |
|                 ^                   |  1  | 0.7           |
| BM1684X/ch_PP-OCRv3_det_fp32.bmodel |  1  | 14.9          |
|                 ^                   |  2  | 57.8          |
| BM1684X/ch_PP-OCRv3_rec_fp32.bmodel |  1  | 1.6           |
|                 ^                   |  2  | 2.8           |
|                 ^                   |  3  | 5.1           |
|                 ^                   |  4  | 9.8           |
| BM1684X/ch_PP-OCRv3_cls_fp32.bmodel |  1  | 0.2           |
|                 ^                   |  2  | 0.3           |
| BM1684X/ch_PP-OCRv3_det_fp16.bmodel |  1  | 3.3           |
|                 ^                   |  2  | 12.3          |
| BM1684X/ch_PP-OCRv3_rec_fp16.bmodel |  1  | 0.8           |
|                 ^                   |  2  | 1.0           |
|                 ^                   |  3  | 1.8           |
|                 ^                   |  4  | 3.4           |
| BM1684X/ch_PP-OCRv3_cls_fp16.bmodel |  1  | 0.1           |
|                 ^                   |  2  | 0.2           |

> **测试说明**：  
> 1. 性能测试结果具有一定的波动性；
> 2. SoC和PCIe的测试结果基本一致。
> 3. `^`符号表示"同上"。

### 7.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看其打印的时间信息。
在不同的测试平台上，使用不同的例程、模型测试`datasets/train_full_images_0`，性能测试结果如下：
|    测试平台  |     测试程序            |        测试模型            |decode_time/crop_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ----------------       | -------------------------- |  --------           | ---------     | ---------     | --------- |
| BM1684 SoC  | ppocr_system_opencv.py | ch_PP-OCRv3_det_fp32.bmodel| 37.96               |  25.51        |  25.08        |  13.04    |
|     ^       |         ^              | ch_PP-OCRv3_rec_fp32.bmodel| 1.67                |  0.60         |  4.11         |  1.37     |
| BM1684 SoC  | ppocr_bmcv.soc         | ch_PP-OCRv3_det_fp32.bmodel| 7.31                |  5.26         |  14.65        |  3.16     |
|     ^       |         ^              | ch_PP-OCRv3_rec_fp32.bmodel| 1.45                |  0.96         |  3.05         |  3.11     |
| BM1684X SoC | ppocr_system_opencv.py | ch_PP-OCRv3_det_fp32.bmodel| 37.79               |  25.99        |  24.90        |  13.50    |
|     ^       |         ^              | ch_PP-OCRv3_rec_fp32.bmodel| 1.67                |  0.59         |  3.08         |  1.59     |
| BM1684X SoC | ppocr_system_opencv.py | ch_PP-OCRv3_det_fp16.bmodel| 37.58               |  25.79        |  14.16        |  13.30    |
|     ^       |         ^              | ch_PP-OCRv3_rec_fp16.bmodel| 1.67                |  0.59         |  2.10         |  1.59     |
| BM1684X SoC | ppocr_bmcv.soc         | ch_PP-OCRv3_det_fp32.bmodel| 6.72                |  1.31         |  13.59        |  3.44     |
|     ^       |         ^              | ch_PP-OCRv3_rec_fp32.bmodel| 2.08                |  0.59         |  1.75         |  3.08     |
| BM1684X SoC | ppocr_bmcv.soc         | ch_PP-OCRv3_det_fp16.bmodel| 6.72                |  1.30         |  2.78         |  3.53     |
|     ^       |         ^              | ch_PP-OCRv3_rec_fp16.bmodel| 2.00                |  0.38         |  0.64         |  3.08     |

> **测试说明**：  
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. BM1684/1684X SoC的主控处理器均为8核 ARM A53 42320 DMIPS @2.3GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异，不同的阈值对后处理时间影响较大。 
> 5. 测试程序串联运行`det`和`rec`模型，`det`模型对应decode_time，`rec`模型对应crop_time。