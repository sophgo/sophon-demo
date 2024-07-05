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
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```

下载的模型包括：
```bash
├── BM1684
│   ├── ch_PP-OCRv3_cls_fp32.bmodel       # 使用TPU-MLIR编译，用于BM1684的FP32 BModel，由batch_size=1和batch_size=4的模型combine得到。
│   ├── ch_PP-OCRv3_det_fp32.bmodel       # 使用TPU-MLIR编译，用于BM1684的FP32 BModel，由batch_size=1和batch_size=4的模型combine得到。
│   ├── ch_PP-OCRv3_rec_fp32.bmodel       # 使用TPU-MLIR编译，用于BM1684的FP32 BModel，由batch_size=1和batch_size=4的模型combine得到。
├── BM1684X      
│   ├── ch_PP-OCRv3_cls_fp16.bmodel       # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，由batch_size=1和batch_size=4的模型combine得到。
│   ├── ch_PP-OCRv3_cls_fp32.bmodel       # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，由batch_size=1和batch_size=4的模型combine得到。
│   ├── ch_PP-OCRv3_det_fp16.bmodel       # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，由batch_size=1和batch_size=4的模型combine得到。
│   ├── ch_PP-OCRv3_det_fp32.bmodel       # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，由batch_size=1和batch_size=4的模型combine得到。
│   ├── ch_PP-OCRv3_rec_fp16.bmodel       # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，由batch_size=1和batch_size=4的模型combine得到。
│   ├── ch_PP-OCRv3_rec_fp32.bmodel       # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，由batch_size=1和batch_size=4的模型combine得到。
│   BM1688      
│   ├── ch_PP-OCRv3_cls_fp16.bmodel       # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，由batch_size=1和batch_size=4且num_core=1的模型combine得到。
│   ├── ch_PP-OCRv3_cls_fp32.bmodel       # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，由batch_size=1和batch_size=4且num_core=1的模型combine得到。
│   ├── ch_PP-OCRv3_det_fp16.bmodel       # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，由batch_size=1和batch_size=4且num_core=1的模型combine得到。
│   ├── ch_PP-OCRv3_det_fp32.bmodel       # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，由batch_size=1和batch_size=4且num_core=1的模型combine得到。
│   ├── ch_PP-OCRv3_rec_fp16.bmodel       # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，由batch_size=1和batch_size=4且num_core=1的模型combine得到。
|   ├── ch_PP-OCRv3_rec_fp32.bmodel       # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，由batch_size=1和batch_size=4且num_core=1的模型combine得到。
│   ├── ch_PP-OCRv3_cls_fp16_2core.bmodel # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，由batch_size=1和batch_size=4且num_core=2的模型combine得到。
│   ├── ch_PP-OCRv3_cls_fp32_2core.bmodel # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，由batch_size=1和batch_size=4且num_core=2的模型combine得到。
│   ├── ch_PP-OCRv3_det_fp16_2core.bmodel # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，由batch_size=1和batch_size=4且num_core=2的模型combine得到。
│   ├── ch_PP-OCRv3_det_fp32_2core.bmodel # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，由batch_size=1和batch_size=4且num_core=2的模型combine得到。
│   ├── ch_PP-OCRv3_rec_fp16_2core.bmodel # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，由batch_size=1和batch_size=4且num_core=2的模型combine得到。
|   └── ch_PP-OCRv3_rec_fp32_2core.bmodel # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，由batch_size=1和batch_size=4且num_core=2的模型combine得到。
├── CV186X      
│   ├── ch_PP-OCRv3_cls_fp16.bmodel       # 使用TPU-MLIR编译，用于CV186X的FP16 BModel，由batch_size=1和batch_size=4的模型combine得到。
│   ├── ch_PP-OCRv3_cls_fp32.bmodel       # 使用TPU-MLIR编译，用于CV186X的FP32 BModel，由batch_size=1和batch_size=4的模型combine得到。
│   ├── ch_PP-OCRv3_det_fp16.bmodel       # 使用TPU-MLIR编译，用于CV186X的FP16 BModel，由batch_size=1和batch_size=4的模型combine得到。
│   ├── ch_PP-OCRv3_det_fp32.bmodel       # 使用TPU-MLIR编译，用于CV186X的FP32 BModel，由batch_size=1和batch_size=4的模型combine得到。
│   ├── ch_PP-OCRv3_rec_fp16.bmodel       # 使用TPU-MLIR编译，用于CV186X的FP16 BModel，由batch_size=1和batch_size=4的模型combine得到。
│   └── ch_PP-OCRv3_rec_fp32.bmodel       # 使用TPU-MLIR编译，用于CV186X的FP32 BModel，由batch_size=1和batch_size=4的模型combine得到。
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

​执行上述命令会在`models/BM1684`等文件夹下生成`ch_PP-OCRv3_det_fp32.bmodel`等文件，即转换好的FP32 BModel。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x #bm1684x/bm1688/cv186x
```

​执行上述命令会在`models/BM1684X/`等文件夹下生成`ch_PP-OCRv3_det_fp16.bmodel`等文件，即转换好的FP16 BModel。

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
| SE5-16       | ppocr_system_opencv.py  | fp32         | 0.575   |
| SE5-16       | ppocr_bmcv.soc          | fp32         | 0.573   |
| SE7-32       | ppocr_system_opencv.py    | fp32       |    0.575 |
| SE7-32       | ppocr_system_opencv.py    | fp16       |    0.565 |
| SE7-32       | ppocr_bmcv.soc            | fp32       |    0.573 |
| SE7-32       | ppocr_bmcv.soc            | fp16       |    0.558 |
| SE9-16       | ppocr_system_opencv.py    | fp32       |    0.575 |
| SE9-16       | ppocr_system_opencv.py    | fp16       |    0.575 |
| SE9-16       | ppocr_bmcv.soc            | fp32       |    0.571 |
| SE9-16       | ppocr_bmcv.soc            | fp16       |    0.571 |
| SE9-8        | ppocr_system_opencv.py    | fp32       |    0.575 |
| SE9-8        | ppocr_system_opencv.py    | fp16       |    0.575 |
| SE9-8        | ppocr_bmcv.soc            | fp32       |    0.571 |
| SE9-8        | ppocr_bmcv.soc            | fp16       |    0.571 |

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
bmrt_test --bmodel models/BM1684X/ch_PP-OCRv3_det_fp32.bmodel
```
测试各个模型的理论推理时间，结果如下：

|                  测试模型                  |stage| calculate time(ms) |
| ----------------------------------        |  ---| ----------------- |
| BM1684/ch_PP-OCRv3_det_fp32.bmodel        |  0  | 15.4          |
|                 ^                         |  1  | 59.7          |
| BM1684/ch_PP-OCRv3_rec_fp32.bmodel        |  0  | 2.7           |
|                 ^                         |  1  | 5.4           |
|                 ^                         |  2  | 8.5           |
|                 ^                         |  3  | 17.1          |
| BM1684/ch_PP-OCRv3_cls_fp32.bmodel        |  0  | 0.3           |
|                 ^                         |  1  | 0.7           |
|    BM1684X/ch_PP-OCRv3_det_fp32.bmodel    |  0  |   14.60       |
|                 ^                         |  1  |   56.89       |
|    BM1684X/ch_PP-OCRv3_rec_fp32.bmodel    |  0  |   1.56        |
|                 ^                         |  1  |   2.70        |
|                 ^                         |  2  |   5.09        |
|                 ^                         |  3  |   10.00       |
|    BM1684X/ch_PP-OCRv3_cls_fp32.bmodel    |  0  |   0.23        |
|                 ^                         |  1  |   0.30        |
|    BM1684X/ch_PP-OCRv3_det_fp16.bmodel    |  0  |   3.10        |
|                 ^                         |  1  |   11.49       |
|    BM1684X/ch_PP-OCRv3_rec_fp16.bmodel    |  0  |   0.65        |
|                 ^                         |  1  |   0.95        |
|                 ^                         |  2  |   1.75        |
|                 ^                         |  3  |   3.34        |
|    BM1684X/ch_PP-OCRv3_cls_fp16.bmodel    |  0  |   0.20        |
|                 ^                         |  1  |   0.16        |
| BM1688/ch_PP-OCRv3_det_fp32.bmodel      |       0 |          50.83  |
| ^                                       |       1 |         207.02  |
| BM1688/ch_PP-OCRv3_rec_fp32.bmodel      |       0 |           8.70  |
| ^                                       |       1 |          16.84  |
| ^                                       |       2 |          33.31  |
| ^                                       |       3 |          62.87  |
| BM1688/ch_PP-OCRv3_cls_fp32.bmodel      |       0 |           0.64  |
| ^                                       |       1 |           1.29  |
| BM1688/ch_PP-OCRv3_det_fp16.bmodel      |       0 |          14.80  |
| ^                                       |       1 |          59.66  |
| BM1688/ch_PP-OCRv3_rec_fp16.bmodel      |       0 |           3.04  |
| ^                                       |       1 |           4.62  |
| ^                                       |       2 |           8.91  |
| ^                                       |       3 |          16.87  |
| BM1688/ch_PP-OCRv3_cls_fp16.bmodel      |       0 |           0.47  |
| ^                                       |       1 |           0.60  |
| BM1688/ch_PP-OCRv3_det_fp32_2core.bmodel|       0 |          36.28  |
| ^                                       |       1 |         123.79  |
| BM1688/ch_PP-OCRv3_rec_fp32_2core.bmodel|       0 |           6.78  |
| ^                                       |       1 |          12.52  |
| ^                                       |       2 |          19.68  |
| ^                                       |       3 |          38.30  |
| BM1688/ch_PP-OCRv3_cls_fp32_2core.bmodel|       0 |           0.65  |
| ^                                       |       1 |           1.20  |
| BM1688/ch_PP-OCRv3_det_fp16_2core.bmodel|       0 |          11.68  |
| ^                                       |       1 |          43.56  |
| BM1688/ch_PP-OCRv3_rec_fp16_2core.bmodel|       0 |           3.04  |
| ^                                       |       1 |           3.98  |
| ^                                       |       2 |           7.01  |
| ^                                       |       3 |          12.82  |
| BM1688/ch_PP-OCRv3_cls_fp16_2core.bmodel|       0 |           0.49  |
| ^                                       |       1 |           0.45  |
| CV186X/ch_PP-OCRv3_det_fp32.bmodel       |       0 |          49.59  |
| ^                                        |       1 |         201.13  |
| CV186X/ch_PP-OCRv3_rec_fp32.bmodel       |       0 |           7.72  |
| ^                                        |       1 |          16.42  |
| ^                                        |       2 |          29.35  |
| ^                                        |       3 |          59.98  |
| CV186X/ch_PP-OCRv3_cls_fp32.bmodel       |       0 |           0.49  |
| ^                                        |       1 |           1.22  |
| CV186X/ch_PP-OCRv3_det_fp16.bmodel       |       0 |          14.31  |
| ^                                        |       1 |          58.67  |
| CV186X/ch_PP-OCRv3_rec_fp16.bmodel       |       0 |           2.10  |
| ^                                        |       1 |           3.78  |
| ^                                        |       2 |           7.19  |
| ^                                        |       3 |          14.23  |
| CV186X/ch_PP-OCRv3_cls_fp16.bmodel       |       0 |           0.30  |
| ^                                        |       1 |           0.53  |

> **测试说明**：  
> 1. 性能测试结果具有一定的波动性；
> 2. SoC和PCIe的测试结果基本一致。
> 3. `^`符号表示"同上"。

### 7.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看其打印的时间信息。
在不同的测试平台上，使用不同的例程、模型测试`datasets/train_full_images_0`，性能测试结果如下：
|    测试平台  |     测试程序             |            测试模型               |decode_time/crop_time|    preprocess_time  |inference_time      |  postprocess_time    | 
| ----------- | ----------------        |     --------------------------    |  --------           | ---------           | ---------           |  ---------          |
| SE5-16      | ppocr_system_opencv.py  |    ch_PP-OCRv3_det_fp32.bmodel    |        37.96        |        25.51        |        25.08        |        13.04        |
|     ^       |         ^               |    ch_PP-OCRv3_rec_fp32.bmodel    |        1.67         |        0.60         |        4.11         |        1.37         |
| SE5-16      | ppocr_bmcv.soc          |    ch_PP-OCRv3_det_fp32.bmodel    |        7.31         |        5.26         |        14.65        |        3.16         |
|     ^       |         ^               |    ch_PP-OCRv3_rec_fp32.bmodel    |        1.45         |        0.96         |        3.05         |        3.11         |
|   SE7-32    | ppocr_system_opencv.py  |    ch_PP-OCRv3_det_fp32.bmodel    |        33.78        |        25.84        |        25.66        |        13.26        |
|      ^      |            ^            |    ch_PP-OCRv3_rec_fp32.bmodel    |        1.66         |        0.59         |        3.22         |        1.60         |
|   SE7-32    | ppocr_system_opencv.py  |    ch_PP-OCRv3_det_fp16.bmodel    |        33.79        |        26.05        |        14.32        |        13.17        |
|      ^      |            ^            |    ch_PP-OCRv3_rec_fp16.bmodel    |        1.67         |        0.60         |        2.18         |        1.60         |
|   SE7-32    |     ppocr_bmcv.soc      |    ch_PP-OCRv3_det_fp32.bmodel    |        5.66         |        1.32         |        14.21        |        3.38         |
|      ^      |            ^            |    ch_PP-OCRv3_rec_fp32.bmodel    |        0.67         |        0.18         |        1.74         |        2.97         |
|   SE7-32    |     ppocr_bmcv.soc      |    ch_PP-OCRv3_det_fp16.bmodel    |        5.66         |        1.32         |        2.87         |        3.38         |
|      ^      |            ^            |    ch_PP-OCRv3_rec_fp16.bmodel    |        0.67         |        0.18         |        0.61         |        2.98         |
|   SE9-16    | ppocr_system_opencv.py  |    ch_PP-OCRv3_det_fp32.bmodel    |        53.49        |        33.75        |        60.79        |        18.13        |
|      ^      |            ^            |    ch_PP-OCRv3_rec_fp32.bmodel    |        2.31         |        0.81         |        11.78        |        1.92         |
|   SE9-16    | ppocr_system_opencv.py  |    ch_PP-OCRv3_det_fp16.bmodel    |        53.15        |        33.64        |        26.31        |        18.06        |
|      ^      |            ^            |    ch_PP-OCRv3_rec_fp16.bmodel    |        2.32         |        0.81         |        4.47         |        1.92         |
|   SE9-16    |     ppocr_bmcv.soc      |    ch_PP-OCRv3_det_fp32.bmodel    |        16.04        |        3.22         |        46.75        |        4.72         |
|      ^      |            ^            |    ch_PP-OCRv3_rec_fp32.bmodel    |        0.80         |        0.61         |        10.42        |        4.16         |
|   SE9-16    |     ppocr_bmcv.soc      |    ch_PP-OCRv3_det_fp16.bmodel    |        16.58        |        3.22         |        11.64        |        4.75         |
|      ^      |            ^            |    ch_PP-OCRv3_rec_fp16.bmodel    |        0.80         |        0.61         |        2.62         |        4.18         |
|   SE9-16    | ppocr_system_opencv.py  | ch_PP-OCRv3_det_fp32_2core.bmodel |        52.67        |        33.78        |        40.89        |        18.09        |
|      ^      |            ^            | ch_PP-OCRv3_rec_fp32_2core.bmodel |        2.32         |        0.80         |        8.01         |        1.92         |
|   SE9-16    | ppocr_system_opencv.py  | ch_PP-OCRv3_det_fp16_2core.bmodel |        53.29        |        33.72        |        22.14        |        18.04        |
|      ^      |            ^            | ch_PP-OCRv3_rec_fp16_2core.bmodel |        2.31         |        0.81         |        3.94         |        1.92         |
|   SE9-16    |     ppocr_bmcv.soc      | ch_PP-OCRv3_det_fp32_2core.bmodel |        16.63        |        3.22         |        25.97        |        4.92         |
|      ^      |            ^            | ch_PP-OCRv3_rec_fp32_2core.bmodel |        0.80         |        0.60         |        6.84         |        4.17         |
|   SE9-16    |     ppocr_bmcv.soc      | ch_PP-OCRv3_det_fp16_2core.bmodel |        17.66        |        3.22         |        7.61         |        4.76         |
|      ^      |            ^            | ch_PP-OCRv3_rec_fp16_2core.bmodel |        0.80         |        0.60         |        2.15         |        4.18         |
|    SE9-8    | ppocr_system_opencv.py  |    ch_PP-OCRv3_det_fp32.bmodel    |        55.13        |        33.78        |        65.30        |        18.14        |
|      ^      |            ^            |    ch_PP-OCRv3_rec_fp32.bmodel    |        2.32         |        0.81         |        11.27        |        1.92         |
|    SE9-8    | ppocr_system_opencv.py  |    ch_PP-OCRv3_det_fp16.bmodel    |        55.00        |        33.83        |        29.85        |        18.10        |
|      ^      |            ^            |    ch_PP-OCRv3_rec_fp16.bmodel    |        2.32         |        0.81         |        4.33         |        1.92         |
|    SE9-8    |     ppocr_bmcv.soc      |    ch_PP-OCRv3_det_fp32.bmodel    |        14.24        |        3.18         |        50.28        |        4.53         |
|      ^      |            ^            |    ch_PP-OCRv3_rec_fp32.bmodel    |        0.76         |        0.55         |        10.03        |        4.16         |
|    SE9-8    |     ppocr_bmcv.soc      |    ch_PP-OCRv3_det_fp16.bmodel    |        14.22        |        3.18         |        14.66        |        4.71         |
|      ^      |            ^            |    ch_PP-OCRv3_rec_fp16.bmodel    |        0.76         |        0.52         |        2.43         |        4.18         |

> **测试说明**：  
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. SE5-16/SE7-32的主控处理器均为8核CA53@2.3GHz，SE9-16为8核CA53@1.6GHz，SE9-8为6核CA53@1.6GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异，不同的阈值对后处理时间影响较大。 
> 5. 测试程序串联运行`det`和`rec`模型，`det`模型对应decode_time，`rec`模型对应crop_time。
