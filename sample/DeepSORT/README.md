# DeepSORT

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
​本例程使用[YOLOv5](../YOLOv5/README.md)中的目标检测模型，并对[Deep Sort with PyTorch](https://github.com/ZQPei/deep_sort_pytorch)的特征提取模型和算法进行移植，使之能在SOPHON BM1684/BM1684X/BM1688上进行推理测试。

## 2. 特性
* 支持BM1688(SoC)/CV186X(SoC)/BM1684X(x86 PCIe、SoC)/BM1684(x86 PCIe、SoC、arm PCIe)
* 支持FP32、FP16(BM1688/BM1684X/CV186X)、INT8模型编译和推理
* 支持基于BMCV预处理的C++推理
* 支持基于OpenCV预处理的Python推理
* 支持单batch和多batch模型推理
* 支持MOT格式数据集(即图片文件夹)和单视频测试
 
## 3. 准备模型与数据
本例程需要准备**目标检测模型**和**特征提取模型**，目标检测模型请参考[YOLOv5](../YOLOv5/README.md#3-准备模型与数据)，下面主要介绍特征提取模型。

建议使用TPU-MLIR编译BModel，Pytorch模型在编译前要导出成onnx模型。`tools/extractor_transform.py`是针对[Deep Sort with PyTorch](https://github.com/ZQPei/deep_sort_pytorch)中模型的转换脚本，可以一次性导出torchscript和onnx模型。**请您根据需要修改代码**。

```
python3 tools/extractor_transform.py --pth_path <your .pth weights>
```

​同时，您需要准备用于测试的数据集或视频，如果量化模型，还要准备用于量化的数据集。

​本例程在`scripts`目录下提供了相关模型和数据的下载脚本`download.sh`，您也可以自己准备模型和数据集，并参考[4. 模型编译](#4-模型编译)进行模型转换。

```bash
# 安装unzip，若已安装请跳过
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```

下载的模型包括：
```bash
./models
├── BM1684
│   ├── extractor_fp32_1b.bmodel              # 使用TPU-MLIR编译，用于BM1684的FP32 BModel，batch_size=1
│   ├── extractor_fp32_4b.bmodel              # 使用TPU-MLIR编译，用于BM1684的FP32 BModel，batch_size=4
│   ├── extractor_int8_1b.bmodel              # 使用TPU-MLIR编译，用于BM1684的INT8 BModel，batch_size=1
│   ├── extractor_int8_4b.bmodel              # 使用TPU-MLIR编译，用于BM1684的INT8 BModel，batch_size=4
│   ├── yolov5s_v6.1_3output_fp32_1b.bmodel   # 从YOLOv5例程中获取，用于BM1684的FP32 BModel，batch_size=1
│   ├── yolov5s_v6.1_3output_int8_1b.bmodel   # 从YOLOv5例程中获取，用于BM1684的INT8 BModel，batch_size=1
│   └── yolov5s_v6.1_3output_int8_4b.bmodel   # 从YOLOv5例程中获取，用于BM1684的INT8 BModel，batch_size=4
├── BM1684X
│   ├── extractor_fp16_1b.bmodel              # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=1
│   ├── extractor_fp16_4b.bmodel              # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=4
│   ├── extractor_fp32_1b.bmodel              # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=1
│   ├── extractor_fp32_4b.bmodel              # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=4
│   ├── extractor_int8_1b.bmodel              # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=1
│   ├── extractor_int8_4b.bmodel              # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=4
│   ├── yolov5s_v6.1_3output_fp16_1b.bmodel   # 从YOLOv5例程中获取，用于BM1684X的FP16 BModel，batch_size=1
│   ├── yolov5s_v6.1_3output_fp32_1b.bmodel   # 从YOLOv5例程中获取，用于BM1684X的FP32 BModel，batch_size=1
│   ├── yolov5s_v6.1_3output_int8_1b.bmodel   # 从YOLOv5例程中获取，用于BM1684X的INT8 BModel，batch_size=1
│   └── yolov5s_v6.1_3output_int8_4b.bmodel   # 从YOLOv5例程中获取，用于BM1684X的INT8 BModel，batch_size=4
├── BM1688
│   ├── extractor_fp16_1b.bmodel              # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1，num_core=1
│   ├── extractor_fp16_4b.bmodel              # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=4，num_core=1
│   ├── extractor_fp32_1b.bmodel              # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1，num_core=1
│   ├── extractor_fp32_4b.bmodel              # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=4，num_core=1
│   ├── extractor_int8_1b.bmodel              # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=1，num_core=1
│   ├── extractor_int8_4b.bmodel              # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=4，num_core=1
│   ├── extractor_fp16_1b_2core.bmodel        # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1，num_core=2
│   ├── extractor_fp16_4b_2core.bmodel        # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=4，num_core=2
│   ├── extractor_fp32_1b_2core.bmodel        # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1，num_core=2
│   ├── extractor_fp32_4b_2core.bmodel        # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=4，num_core=2
│   ├── extractor_int8_1b_2core.bmodel        # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=1，num_core=2
│   ├── extractor_int8_4b_2core.bmodel        # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=4，num_core=2
│   └── yolov5s_v6.1_3output_int8_1b.bmodel   # 从YOLOv5例程中获取，用于BM1688的INT8 BModel，batch_size=1
├── CV186X
│   ├── extractor_fp16_1b.bmodel               # 使用TPU-MLIR编译，用于CV186X的FP16 BModel，batch_size=1
│   ├── extractor_fp16_4b.bmodel               # 使用TPU-MLIR编译，用于CV186X的FP16 BModel，batch_size=1
│   ├── extractor_fp32_1b.bmodel               # 使用TPU-MLIR编译，用于CV186X的FP16 BModel，batch_size=1
│   ├── extractor_fp32_4b.bmodel               # 使用TPU-MLIR编译，用于CV186X的FP16 BModel，batch_size=1
│   ├── extractor_int8_1b.bmodel               # 使用TPU-MLIR编译，用于CV186X的FP16 BModel，batch_size=1
│   ├── extractor_int8_4b.bmodel               # 使用TPU-MLIR编译，用于CV186X的FP16 BModel，batch_size=1
│   └── yolov5s_v6.1_3output_int8_1b.bmodel    # 从YOLOv5例程中获取，用于CV186X的INT8 BModel，batch_size=1
├── onnx
│   └── extractor.onnx                        # 由ckpt.t7导出的onnx模型
└── torch
    └── extractor.pt                          # 由ckpt.t7导出的torchscript模型
```
下载的数据包括：
```
./datasets
├── cali_set                                  # 量化数据集
├── test_car_person_1080P.mp4                 # 测试视频
└── mot15_trainset                            # MOT15的训练集，这里用于评价指标测试。 
```

## 4. 模型编译
导出的模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。建议使用TPU-MLIR编译BModel。

模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的SDK中获取)。

- 生成FP32 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684/BM1684X/BM1688**），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684 #bm1684x/bm1688/cv186x
```

​执行上述命令会在`models/BM1684`等文件夹下生成`extractor_fp32_1b.bmodel`文件，即转换好的FP32 BModel。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688**），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x #bm1688/cv186x
```

​执行上述命令会在`models/BM1684X/`等文件夹下生成`extractor_fp16_1b.bmodel`文件，即转换好的FP16 BModel。

- 生成INT8 BModel

​本例程在`scripts`目录下提供了量化INT8 BModel的脚本，请注意修改`gen_int8bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台（**支持BM1684/BM1684X/BM1688**），如：

```shell
./scripts/gen_int8bmodel_mlir.sh bm1684 #bm1684x/bm1688/cv186x
```

​上述脚本会在`models/BM1684`等文件夹下生成`extractor_int8_1b.bmodel`等文件，即转换好的INT8 BModel。



## 5. 例程测试
- [C++例程](./cpp/README.md)
- [Python例程](./python/README.md)

## 6. 精度测试
### 6.1 测试方法

首先，参考[C++例程](cpp/README.md#32-测试MOT数据集)或[Python例程](python/README.md#22-测试MOT数据集)推理要测试的数据集，生成包含目标追踪结果的txt文件，注意修改数据集(datasets/mot15_trainset/ADL-Rundle-6/img1)。  
然后，使用`tools`目录下的`eval_mot15.py`脚本，将测试生成的txt文件与测试集标签txt文件进行对比，计算出目标追踪的一系列评价指标，命令如下：
```bash
# 安装motmetrics，若已安装请跳过
pip3 install motmetrics
# 请根据实际情况修改程序路径和txt文件路径
python3 tools/eval_mot15.py --gt_file datasets/mot15_trainset/ADL-Rundle-6/gt/gt.txt --ts_file python/results/mot_eval/ADL-Rundle-6_extractor_fp32_1b.bmodel.txt
```
运行结果：
```bash
MOTA = 0.43801157915751643
     num_frames      IDF1       IDP       IDR      Rcll      Prcn    GT  MT  PT  ML    FP    FN  IDsw  FM      MOTA      MOTP
acc         525  0.524889  0.544908  0.506289  0.687163  0.739579  5009  10  12   2  1212  1567    36  79  0.438012  0.218005
```
### 6.2 测试结果
这里使用目标检测模型`yolov5s_v6.1_3output_int8_1b.bmodel`，使用数据集ADL-Rundle-6，记录MOTA作为精度指标，精度测试结果如下：
|   测试平台    |      测试程序     |           测试模型         | MOTA |
| ------------ | ---------------- | -------------------------- | ---- |
|    SE5-16    | deepsort_opencv.py | extractor_fp32_1b.bmodel | 0.457 |
|    SE5-16    | deepsort_opencv.py | extractor_int8_1b.bmodel | 0.459 |
|    SE5-16    | deepsort_bmcv.pcie | extractor_fp32_1b.bmodel | 0.450 |
|    SE5-16    | deepsort_bmcv.pcie | extractor_int8_1b.bmodel | 0.452 |
|    SE7-32    | deepsort_opencv.py | extractor_fp32_1b.bmodel | 0.439 |
|    SE7-32    | deepsort_opencv.py | extractor_fp16_1b.bmodel | 0.439 |
|    SE7-32    | deepsort_opencv.py | extractor_int8_1b.bmodel | 0.436 |
|    SE7-32    | deepsort_bmcv.pcie | extractor_fp32_1b.bmodel | 0.442 |
|    SE7-32    | deepsort_bmcv.pcie | extractor_fp16_1b.bmodel | 0.442 |
|    SE7-32    | deepsort_bmcv.pcie | extractor_int8_1b.bmodel | 0.437 |
|    SE9-16    | deepsort_opencv.py | extractor_fp32_1b.bmodel | 0.441 |
|    SE9-16    | deepsort_opencv.py | extractor_fp16_1b.bmodel | 0.441 |
|    SE9-16    | deepsort_opencv.py | extractor_int8_1b.bmodel | 0.440 |
|    SE9-16    | deepsort_bmcv.soc  | extractor_fp32_1b.bmodel | 0.430 |
|    SE9-16    | deepsort_bmcv.soc  | extractor_fp16_1b.bmodel | 0.430 |
|    SE9-16    | deepsort_bmcv.soc  | extractor_int8_1b.bmodel | 0.429 |
|    SE9-8     | deepsort_opencv.py | extractor_fp32_1b.bmodel | 0.441 |
|    SE9-8     | deepsort_opencv.py | extractor_fp16_1b.bmodel | 0.441 |
|    SE9-8     | deepsort_opencv.py | extractor_int8_1b.bmodel | 0.440 |
|    SE9-8     | deepsort_bmcv.soc  | extractor_fp32_1b.bmodel | 0.429 |
|    SE9-8     | deepsort_bmcv.soc  | extractor_fp16_1b.bmodel | 0.430 |
|    SE9-8     | deepsort_bmcv.soc  | extractor_int8_1b.bmodel | 0.429 |

> **测试说明**：  
> 1. batch_size=4和batch_size=1的模型精度一致；
> 2. 由于sdk版本之间可能存在差异，实际运行结果与本表有<1%的精度误差是正常的；
> 3. BM1688 num_core=2的模型与num_core=1的模型精度基本一致。
> 4. 在搭载了相同TPU和SOPHONSDK的PCIe或SoC平台上，相同程序的精度一致，SE5系列对应BM1684，SE7系列对应BM1684X，SE9系列中，SE9-16对应BM1688，SE9-8对应CV186X；

## 7. 性能测试
### 7.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684X/extractor_fp32_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

|           测试模型                    | calculate time(ms) |
| -----------------------------         | ----------------- |
| BM1684/extractor_fp32_1b.bmodel       |   2.26        |
| BM1684/extractor_fp32_4b.bmodel       |   1.25        |
| BM1684/extractor_int8_1b.bmodel       |   0.99        |
| BM1684/extractor_int8_4b.bmodel       |   0.25        |
| BM1684X/extractor_fp32_1b.bmodel      |   2.08        |
| BM1684X/extractor_fp32_4b.bmodel      |   1.88        |
| BM1684X/extractor_fp16_1b.bmodel      |   0.56        |
| BM1684X/extractor_fp16_4b.bmodel      |   0.24        |
| BM1684X/extractor_int8_1b.bmodel      |   0.33        |
| BM1684X/extractor_int8_4b.bmodel      |   0.14        |
| BM1688/extractor_fp32_1b.bmodel       |   13.29       |
| BM1688/extractor_fp32_4b.bmodel       |   11.27       |
| BM1688/extractor_fp16_1b.bmodel       |   3.14        |
| BM1688/extractor_fp16_4b.bmodel       |   1.84        |
| BM1688/extractor_int8_1b.bmodel       |   1.93        |
| BM1688/extractor_int8_4b.bmodel       |   0.75        |
| BM1688/extractor_fp32_1b_2core.bmodel |   13.34       |
| BM1688/extractor_fp32_4b_2core.bmodel |   6.36        |
| BM1688/extractor_fp16_1b_2core.bmodel |   3.49        |
| BM1688/extractor_fp16_4b_2core.bmodel |   1.32        |
| BM1688/extractor_int8_1b_2core.bmodel |   1.87        |
| BM1688/extractor_int8_4b_2core.bmodel |   0.75        |
| CV186X/extractor_fp32_1b.bmodel       |   11.13       |
| CV186X/extractor_fp32_4b.bmodel       |   10.78       |
| CV186X/extractor_fp16_1b.bmodel       |    2.57       |
| CV186X/extractor_fp16_4b.bmodel       |    1.48       |
| CV186X/extractor_int8_1b.bmodel       |    1.20       |
| CV186X/extractor_int8_4b.bmodel       |    0.55       |

> **测试说明**：  
1. 性能测试结果具有一定的波动性；
2. `calculate time`已折算为平均每张图片的推理时间。
3. SoC和PCIe的测试结果基本一致。

### 7.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。这里**只统计特征提取模型的时间**，解码、目标检测模型的时间请参考[YOLOV5](../YOLOv5/README.md#72-程序运行性能)。

这里使用目标检测模型`yolov5s_v6.1_3output_int8_1b.bmodel`，在不同的测试平台上，使用不同的例程、模型测试`datasets/mot15_trainset/ADL-Rundle-6/img1`，性能测试结果如下：
|  测试平台   |      测试程序      |              测试模型             | preprocess_time | inference_time  |postprocess_time |
|-------------|-------------------|-----------------------------------|-----------------|-----------------|-----------------|
|   SE7-32    |deepsort_opencv.py |     extractor_fp32_1b.bmodel      |      2.19       |      3.11       |      94.35      |
|   SE7-32    |deepsort_opencv.py |     extractor_fp32_4b.bmodel      |      2.19       |      2.66       |     148.60      |
|   SE7-32    |deepsort_opencv.py |     extractor_fp16_1b.bmodel      |      2.21       |      1.48       |      75.69      |
|   SE7-32    |deepsort_opencv.py |     extractor_fp16_4b.bmodel      |      2.15       |      0.76       |      66.93      |
|   SE7-32    |deepsort_opencv.py |     extractor_int8_1b.bmodel      |      2.17       |      1.27       |      64.47      |
|   SE7-32    |deepsort_opencv.py |     extractor_int8_4b.bmodel      |      2.14       |      0.65       |      75.33      |
|   SE7-32    | deepsort_bmcv.soc |     extractor_fp32_1b.bmodel      |      0.14       |      2.11       |      4.93       |
|   SE7-32    | deepsort_bmcv.soc |     extractor_fp32_4b.bmodel      |      0.36       |      7.29       |      5.54       |
|   SE7-32    | deepsort_bmcv.soc |     extractor_fp16_1b.bmodel      |      0.13       |      0.51       |      5.25       |
|   SE7-32    | deepsort_bmcv.soc |     extractor_fp16_4b.bmodel      |      0.35       |      0.93       |      5.48       |
|   SE7-32    | deepsort_bmcv.soc |     extractor_int8_1b.bmodel      |      0.13       |      0.28       |      5.82       |
|   SE7-32    | deepsort_bmcv.soc |     extractor_int8_4b.bmodel      |      0.35       |      0.54       |      5.74       |
|   SE9-16    |deepsort_opencv.py |     extractor_fp32_1b.bmodel      |      3.05       |      13.57      |      79.64      |
|   SE9-16    |deepsort_opencv.py |     extractor_fp32_4b.bmodel      |      3.01       |      13.54      |      75.79      |
|   SE9-16    |deepsort_opencv.py |     extractor_fp16_1b.bmodel      |      3.03       |      3.36       |      74.38      |
|   SE9-16    |deepsort_opencv.py |     extractor_fp16_4b.bmodel      |      3.01       |      2.44       |      76.62      |
|   SE9-16    |deepsort_opencv.py |     extractor_int8_1b.bmodel      |      3.04       |      2.08       |      82.18      |
|   SE9-16    |deepsort_opencv.py |     extractor_int8_4b.bmodel      |      3.00       |      1.20       |      74.43      |
|   SE9-16    | deepsort_bmcv.soc |     extractor_fp32_1b.bmodel      |      0.47       |      12.26      |      6.70       |
|   SE9-16    | deepsort_bmcv.soc |     extractor_fp32_4b.bmodel      |      1.44       |      43.84      |      6.59       |
|   SE9-16    | deepsort_bmcv.soc |     extractor_fp16_1b.bmodel      |      0.43       |      2.08       |      6.57       |
|   SE9-16    | deepsort_bmcv.soc |     extractor_fp16_4b.bmodel      |      1.42       |      6.17       |      6.60       |
|   SE9-16    | deepsort_bmcv.soc |     extractor_int8_1b.bmodel      |      0.44       |      0.81       |      6.68       |
|   SE9-16    | deepsort_bmcv.soc |     extractor_int8_4b.bmodel      |      1.43       |      1.95       |      6.40       |
|    SE9-8    |deepsort_opencv.py |     extractor_fp32_1b.bmodel      |      3.05       |      12.25      |      55.49      |
|    SE9-8    |deepsort_opencv.py |     extractor_fp32_4b.bmodel      |      3.01       |      13.28      |      56.58      |
|    SE9-8    |deepsort_opencv.py |     extractor_fp16_1b.bmodel      |      3.03       |      3.68       |      47.59      |
|    SE9-8    |deepsort_opencv.py |     extractor_fp16_4b.bmodel      |      2.99       |      2.32       |      51.97      |
|    SE9-8    |deepsort_opencv.py |     extractor_int8_1b.bmodel      |      3.03       |      2.33       |      48.70      |
|    SE9-8    |deepsort_opencv.py |     extractor_int8_4b.bmodel      |      2.98       |      1.23       |      51.26      |
|    SE9-8    | deepsort_bmcv.soc |     extractor_fp32_1b.bmodel      |      0.44       |      10.97      |      6.76       |
|    SE9-8    | deepsort_bmcv.soc |     extractor_fp32_4b.bmodel      |      1.42       |      42.98      |      6.79       |
|    SE9-8    | deepsort_bmcv.soc |     extractor_fp16_1b.bmodel      |      0.43       |      2.42       |      6.64       |
|    SE9-8    | deepsort_bmcv.soc |     extractor_fp16_4b.bmodel      |      1.41       |      5.80       |      6.79       |
|    SE9-8    | deepsort_bmcv.soc |     extractor_int8_1b.bmodel      |      0.42       |      1.06       |      6.59       |
|    SE9-8    | deepsort_bmcv.soc |     extractor_int8_4b.bmodel      |      1.39       |      2.06       |      6.57       |

> **测试说明**：  
1. 时间单位均为毫秒(ms)，preprocess_time、inference_time是特征提取模型平均每个crop的处理时间，postprocess_time是deepsort算法平均每帧的后处理时间；
2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
3. SE5-16/SE7-32的主控处理器均为8核CA53@2.3GHz，SE9-16为8核CA53@1.6GHz，SE9-8为6核CA53@1.6GHz，PCIe上的性能由于处理器的不同可能存在较大差异；

## 8. FAQ
请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。