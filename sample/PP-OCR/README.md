# PP-OCR

## 目录

- [PP-OCR](#pp-ocr)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
  - [3. 准备模型与数据](#3-准备模型与数据)
  - [4. 模型编译](#4-模型编译)
  - [5. 推理测试](#5-推理测试)
  - [6. 精度测试](#6-精度测试)
    - [6.1 测试方法](#61-测试方法)
    - [6.2 测试结果](#62-测试结果)
  - [7. 性能测试](#7-性能测试)
    - [7.1 bmrt\_test](#71-bmrt_test)
    - [7.2 程序运行性能](#72-程序运行性能)
    
## 1. 简介

PP-OCRv4，是百度飞桨团队开源的超轻量OCR系列模型，包含文本检测、文本分类、文本识别模型，是PaddleOCR工具库的重要组成之一。支持中英文数字组合识别、竖排文本识别、长文本识别，其性能及精度较之前的PP-OCR版本均有明显提升。本例程对[PaddleOCR-release-2.8](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.8)的`ch_PP-OCRv4_xx`系列模型和算法进行移植，使之能在SOPHON BM1684/BM1684X/BM1688上进行推理测试。

## 2. 特性
* 支持BM1688/CV186X(SoC)、BM1684X(x86 PCIe、SoC)、BM1684(x86 PCIe、SoC、arm PCIe)
* 支持FP32、FP16(BM1684X/BM1688/CV186X)模型编译和推理
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
chmod -R +x scripts/
./scripts/download.sh
```

下载的模型包括：
```bash
├── BM1684
│   ├── ch_PP-OCRv4_det_fp32.bmodel       # 使用TPU-MLIR编译，用于BM1684的FP32 BModel，由batch_size=1和batch_size=4的模型combine得到。
│   ├── ch_PP-OCRv4_rec_fp32.bmodel       # 使用TPU-MLIR编译，用于BM1684的FP32 BModel，由batch_size=1和batch_size=4的模型combine得到。
├── BM1684X      
│   ├── ch_PP-OCRv4_det_fp16.bmodel       # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，由batch_size=1和batch_size=4的模型combine得到。
│   ├── ch_PP-OCRv4_det_fp32.bmodel       # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，由batch_size=1和batch_size=4的模型combine得到。
│   ├── ch_PP-OCRv4_rec_fp16.bmodel       # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，由batch_size=1和batch_size=4的模型combine得到。
│   ├── ch_PP-OCRv4_rec_fp32.bmodel       # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，由batch_size=1和batch_size=4的模型combine得到。
│   BM1688      
│   ├── ch_PP-OCRv4_det_fp16.bmodel       # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，由batch_size=1和batch_size=4且num_core=1的模型combine得到。
│   ├── ch_PP-OCRv4_det_fp32.bmodel       # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，由batch_size=1和batch_size=4且num_core=1的模型combine得到。
│   ├── ch_PP-OCRv4_rec_fp16.bmodel       # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，由batch_size=1和batch_size=4且num_core=1的模型combine得到。
|   ├── ch_PP-OCRv4_rec_fp32.bmodel       # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，由batch_size=1和batch_size=4且num_core=1的模型combine得到。
│   ├── ch_PP-OCRv4_det_fp16_2core.bmodel # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，由batch_size=1和batch_size=4且num_core=2的模型combine得到。
│   ├── ch_PP-OCRv4_det_fp32_2core.bmodel # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，由batch_size=1和batch_size=4且num_core=2的模型combine得到。
│   ├── ch_PP-OCRv4_rec_fp16_2core.bmodel # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，由batch_size=1和batch_size=4且num_core=2的模型combine得到。
|   └── ch_PP-OCRv4_rec_fp32_2core.bmodel # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，由batch_size=1和batch_size=4且num_core=2的模型combine得到。
├── CV186X      
│   ├── ch_PP-OCRv4_det_fp16.bmodel       # 使用TPU-MLIR编译，用于CV186X的FP16 BModel，由batch_size=1和batch_size=4的模型combine得到。
│   ├── ch_PP-OCRv4_det_fp32.bmodel       # 使用TPU-MLIR编译，用于CV186X的FP32 BModel，由batch_size=1和batch_size=4的模型combine得到。
│   ├── ch_PP-OCRv4_rec_fp16.bmodel       # 使用TPU-MLIR编译，用于CV186X的FP16 BModel，由batch_size=1和batch_size=4的模型combine得到。
│   └── ch_PP-OCRv4_rec_fp32.bmodel       # 使用TPU-MLIR编译，用于CV186X的FP32 BModel，由batch_size=1和batch_size=4的模型combine得到。
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

模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的SDK中获取)。

- 生成FP32 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684/BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684 #bm1684x/bm1688/cv186x
```

​执行上述命令会在`models/BM1684`等文件夹下生成`ch_PP-OCRv4_det_fp32.bmodel`等文件，即转换好的FP32 BModel。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x #bm1684x/bm1688/cv186x
```

​执行上述命令会在`models/BM1684X/`等文件夹下生成`ch_PP-OCRv4_det_fp16.bmodel`等文件，即转换好的FP16 BModel。

- 本例程暂时不支持量化。

## 5. 推理测试
* [C++例程](cpp/README.md)
* [Python例程](python/README.md)


## 6. 精度测试
### 6.1 测试方法
当你运行了[C++推理](cpp/README.md#3-推理测试)或[Python推理](python/README.md#24-全流程推理测试)，生成了结果文件之后，你可以使用本例程的精度测试脚本`tools/eval_icdar.py`和Ground-truth文件`datasets/train_full_images_0.json`来计算结果文件的`F-score/precision/recall`，测试命令如下：

```bash
pip3 install -r python/requirements.txt #如果已经装好不必再装。
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
| SE7-32       | ppocr_system_opencv.py    | fp32       |    0.608 |
| SE7-32       | ppocr_system_opencv.py    | fp16       |    0.608 |
| SE7-32       | ppocr_bmcv.soc            | fp32       |    0.604 |
| SE7-32       | ppocr_bmcv.soc            | fp16       |    0.604 |
| SE9-16       | ppocr_system_opencv.py    | fp32       |    0.608 |
| SE9-16       | ppocr_system_opencv.py    | fp16       |    0.608 |
| SE9-16       | ppocr_bmcv.soc            | fp32       |    0.604 |
| SE9-16       | ppocr_bmcv.soc            | fp16       |    0.604 |
| SE9-8        | ppocr_system_opencv.py    | fp32       |    0.608 |
| SE9-8        | ppocr_system_opencv.py    | fp16       |    0.608 |
| SE9-8        | ppocr_bmcv.soc            | fp32       |    0.605 |
| SE9-8        | ppocr_bmcv.soc            | fp16       |    0.604 |

> **测试说明**：  
> 1. 模型精度为fp32(fp16)，即代表检测模型和识别模型都是fp32(fp16)的精度；
> 2. 由于sdk版本之间可能存在差异，实际运行结果与本表有<0.01的精度误差是正常的；
> 3. 在搭载了相同TPU和SOPHONSDK的PCIe或SoC平台上，相同程序的精度一致，SE5系列对应BM1684，SE7系列对应BM1684X，SE9系列中，SE9-16对应BM1688，SE9-8对应CV186X；
> 4. BM1688 num_core=2的模型与num_core=1的模型精度基本一致；

## 7. 性能测试
### 7.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684X/ch_PP-OCRv4_det_fp32.bmodel
```
测试各个模型的理论推理时间，结果如下：

|                  测试模型                  |stage| calculate time(ms) |
| ----------------------------------        |  ---| ----------------- |
| BM1684X/ch_PP-OCRv4_det_fp32.bmodel      |       0 |          15.50  |
| ^                                        |       1 |          60.75  |
| BM1684X/ch_PP-OCRv4_rec_fp32.bmodel      |       0 |           1.62  |
| ^                                        |       1 |           2.88  |
| ^                                        |       2 |           5.37  |
| ^                                        |       3 |          10.78  |
| BM1684X/ch_PP-OCRv4_det_fp16.bmodel      |       0 |           3.80  |
| ^                                        |       1 |          14.38  |
| BM1684X/ch_PP-OCRv4_rec_fp16.bmodel      |       0 |           0.60  |
| ^                                        |       1 |           0.87  |
| ^                                        |       2 |           1.61  |
| ^                                        |       3 |           2.99  |
| BM1688/ch_PP-OCRv4_det_fp32.bmodel       |       0 |          48.66  |
| ^                                        |       1 |         196.37  |
| BM1688/ch_PP-OCRv4_rec_fp32.bmodel       |       0 |           8.20  |
| ^                                        |       1 |          16.48  |
| ^                                        |       2 |          31.11  |
| ^                                        |       3 |          62.05  |
| BM1688/ch_PP-OCRv4_det_fp16.bmodel       |       0 |          12.12  |
| ^                                        |       1 |          48.61  |
| BM1688/ch_PP-OCRv4_rec_fp16.bmodel       |       0 |           2.07  |
| ^                                        |       1 |           3.86  |
| ^                                        |       2 |           6.95  |
| ^                                        |       3 |          14.39  |
| BM1688/ch_PP-OCRv4_det_fp32_2core.bmodel |       0 |          29.91  |
| ^                                        |       1 |         106.49  |
| BM1688/ch_PP-OCRv4_rec_fp32_2core.bmodel |       0 |           5.83  |
| ^                                        |       1 |          10.06  |
| ^                                        |       2 |          17.63  |
| ^                                        |       3 |          34.48  |
| BM1688/ch_PP-OCRv4_det_fp16_2core.bmodel |       0 |           8.22  |
| ^                                        |       1 |          31.43  |
| BM1688/ch_PP-OCRv4_rec_fp16_2core.bmodel |       0 |           2.07  |
| ^                                        |       1 |           3.23  |
| ^                                        |       2 |           4.46  |
| ^                                        |       3 |           9.41  |
| CV186X/ch_PP-OCRv4_det_fp32.bmodel       |       0 |          48.58  |
| ^                                        |       1 |         196.22  |
| CV186X/ch_PP-OCRv4_rec_fp32.bmodel       |       0 |           8.12  |
| ^                                        |       1 |          16.48  |
| ^                                        |       2 |          31.10  |
| ^                                        |       3 |          61.96  |
| CV186X/ch_PP-OCRv4_det_fp16.bmodel       |       0 |          12.04  |
| ^                                        |       1 |          48.46  |
| CV186X/ch_PP-OCRv4_rec_fp16.bmodel       |       0 |           2.00  |
| ^                                        |       1 |           3.85  |
| ^                                        |       2 |           6.96  |
| ^                                        |       3 |          14.42  |

> **测试说明**：  
> 1. 性能测试结果具有一定的波动性；
> 2. SoC和PCIe的测试结果基本一致。
> 3. `^`符号表示"同上"。

### 7.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看其打印的时间信息。
在不同的测试平台上，使用不同的例程、模型测试`datasets/train_full_images_0`，性能测试结果如下：
|    测试平台  |     测试程序             |            测试模型               |decode_time/crop_time|    preprocess_time  |inference_time      |  postprocess_time    | 
| ----------- | ----------------        |     --------------------------    |  --------           | ---------           | ---------           |  ---------          |
|   SE7-32    | ppocr_system_opencv.py  |    ch_PP-OCRv4_det_fp32.bmodel    |        22.98        |        25.63        |        26.76        |        13.62        |
|      ^      |            ^            |    ch_PP-OCRv4_rec_fp32.bmodel    |        1.59         |        0.56         |        3.20         |        1.55         |
|   SE7-32    | ppocr_system_opencv.py  |    ch_PP-OCRv4_det_fp16.bmodel    |        23.00        |        25.62        |        15.20        |        13.55        |
|      ^      |            ^            |    ch_PP-OCRv4_rec_fp16.bmodel    |        1.60         |        0.56         |        2.07         |        1.55         |
|   SE7-32    |     ppocr_bmcv.soc      |    ch_PP-OCRv4_det_fp32.bmodel    |        14.34        |        1.26         |        15.19        |        3.48         |
|      ^      |            ^            |    ch_PP-OCRv4_rec_fp32.bmodel    |        0.68         |        0.18         |        1.81         |        2.91         |
|   SE7-32    |     ppocr_bmcv.soc      |    ch_PP-OCRv4_det_fp16.bmodel    |        14.44        |        1.26         |        3.60         |        3.56         |
|      ^      |            ^            |    ch_PP-OCRv4_rec_fp16.bmodel    |        0.68         |        0.18         |        0.56         |        2.91         |
|   SE9-16    | ppocr_system_opencv.py  |    ch_PP-OCRv4_det_fp32.bmodel    |        27.23        |        33.41        |        65.26        |        19.00        |
|      ^      |            ^            |    ch_PP-OCRv4_rec_fp32.bmodel    |        2.25         |        0.77         |        11.30        |        1.86         |
|   SE9-16    | ppocr_system_opencv.py  |    ch_PP-OCRv4_det_fp16.bmodel    |        26.85        |        33.61        |        27.85        |        18.78        |
|      ^      |            ^            |    ch_PP-OCRv4_rec_fp16.bmodel    |        2.24         |        0.77         |        4.16         |        1.87         |
|   SE9-16    |     ppocr_bmcv.soc      |    ch_PP-OCRv4_det_fp32.bmodel    |        14.42        |        3.10         |        49.08        |        4.46         |
|      ^      |            ^            |    ch_PP-OCRv4_rec_fp32.bmodel    |        0.85         |        0.41         |        10.14        |        4.07         |
|   SE9-16    |     ppocr_bmcv.soc      |    ch_PP-OCRv4_det_fp16.bmodel    |        14.90        |        3.10         |        12.14        |        4.36         |
|      ^      |            ^            |    ch_PP-OCRv4_rec_fp16.bmodel    |        0.84         |        0.41         |        2.34         |        4.08         |
|   SE9-16    | ppocr_system_opencv.py  | ch_PP-OCRv4_det_fp32_2core.bmodel |        27.24        |        33.46        |        42.42        |        18.77        |
|      ^      |            ^            | ch_PP-OCRv4_rec_fp32_2core.bmodel |        2.24         |        0.76         |        7.39         |        1.86         |
|   SE9-16    | ppocr_system_opencv.py  | ch_PP-OCRv4_det_fp16_2core.bmodel |        27.26        |        33.49        |        23.29        |        18.76        |
|      ^      |            ^            | ch_PP-OCRv4_rec_fp16_2core.bmodel |        2.25         |        0.76         |        3.50         |        1.86         |
|   SE9-16    |     ppocr_bmcv.soc      | ch_PP-OCRv4_det_fp32_2core.bmodel |        14.56        |        3.10         |        26.65        |        4.39         |
|      ^      |            ^            | ch_PP-OCRv4_rec_fp32_2core.bmodel |        0.85         |        0.41         |        6.08         |        4.07         |
|   SE9-16    |     ppocr_bmcv.soc      | ch_PP-OCRv4_det_fp16_2core.bmodel |        15.04        |        3.10         |        7.85         |        4.43         |
|      ^      |            ^            | ch_PP-OCRv4_rec_fp16_2core.bmodel |        0.84         |        0.41         |        1.79         |        4.08         |
|    SE9-8    | ppocr_system_opencv.py  |    ch_PP-OCRv4_det_fp32.bmodel    |        27.23        |        34.28        |        64.38        |        19.04        |
|      ^      |            ^            |    ch_PP-OCRv4_rec_fp32.bmodel    |        2.22         |        0.78         |        11.33        |        1.89         |
|    SE9-8    | ppocr_system_opencv.py  |    ch_PP-OCRv4_det_fp16.bmodel    |        27.17        |        34.08        |        27.46        |        18.98        |
|      ^      |            ^            |    ch_PP-OCRv4_rec_fp16.bmodel    |        2.21         |        0.77         |        4.18         |        1.89         |
|    SE9-8    |     ppocr_bmcv.soc      |    ch_PP-OCRv4_det_fp32.bmodel    |        14.18        |        3.09         |        49.05        |        4.60         |
|      ^      |            ^            |    ch_PP-OCRv4_rec_fp32.bmodel    |        0.85         |        0.40         |        10.15        |        4.05         |
|    SE9-8    |     ppocr_bmcv.soc      |    ch_PP-OCRv4_det_fp16.bmodel    |        14.21        |        3.09         |        12.12        |        4.66         |
|      ^      |            ^            |    ch_PP-OCRv4_rec_fp16.bmodel    |        0.84         |        0.40         |        2.34         |        4.06         |

> **测试说明**：  
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. SE5-16/SE7-32的主控处理器均为8核CA53@2.3GHz，SE9-16为8核CA53@1.6GHz，SE9-8为6核CA53@1.6GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异，不同的阈值对后处理时间影响较大。 
> 5. 测试程序串联运行`det`和`rec`模型，`det`模型对应decode_time，`rec`模型对应crop_time。
