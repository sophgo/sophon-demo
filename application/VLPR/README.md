[简体中文](./README.md)

# VLPR(Vehicle License Plate Recognition)

## 目录

- [VLPR(Vehicle License Plate Recognition)](#vlprvehicle-license-plate-recognition)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
  - [3. 准备模型与数据](#3-准备模型与数据)
  - [4. 模型编译](#4-模型编译)
  - [5. 例程测试](#5-例程测试)
  - [6. 性能测试](#6-性能测试)
    - [6.1 bmrt\_test](#61-bmrt_test)
    - [6.2 程序运行性能](#62-程序运行性能)
  - [7. FAQ](#7-faq)
  
## 1. 简介
本例程对[LPRNet_Pytorch](https://github.com/sirius-ai/LPRNet_Pytorch)的模型和算法进行移植，，使之能够完成全流程车牌检测识别业务，并在SOPHON BM1684/BM1684X/BM1688上进行推理测试。在您使用本例程之前，推荐先跑通[LPRNet](../../sample/LPRNet/README.md)和[YoLov5](../../sample/YOLOv5/README.md)

**论文:** [LPRNet: License Plate Recognition via Deep Neural Networks](https://arxiv.org/abs/1806.10447v1)
**LPRNET 车牌检测源代码**(https://github.com/sirius-ai/LPRNet_Pytorch)

## 2. 特性
* 支持BM1688/CV186X(SoC)、BM1684X(x86 PCIe、SoC)、BM1684(x86 PCIe、SoC、arm PCIe)
* LPRNet支持FP32、FP16(BM1684X/BM1688/CV186X)、INT8模型编译和推理
* YOLOv5支持FP32、INT8模型编译和推理
* 支持C++和Python
* 支持单batch和多batch模型推理
* 支持图片和视频测试
* pipeline式demo，支持解码、预处理和后处理多线程和推理线程并行运行，更充分利用硬件加速

## 3. 准备模型与数据
​本例程在`scripts`目录下提供了相关模型和数据的下载脚本`download.sh`，您也可以自己准备模型和数据集，并参考[4. 模型编译](#4-模型编译)进行模型转换。

```bash
chmod -R +x scripts/
./scripts/download.sh

```
执行后，模型保存至`models/`，数据集下载并解压至`datasets/`
```
下载的模型包括：
./models
├── lprnet
|   ├── BM1684
|   │   ├── lprnet_fp32_1b.bmodel                                     # 使用TPU-MLIR编译，用于BM1684的FP32 LPRNet BModel，batch_size=1，num_core=1
|   │   ├── lprnet_int8_1b.bmodel                                     # 使用TPU-MLIR编译，用于BM1684的INT8 LPRNet BModel，batch_size=1，num_core=1
|   │   └── lprnet_int8_4b.bmodel                                     # 使用TPU-MLIR编译，用于BM1684的INT8 LPRNet BModel，batch_size=4，num_core=1
|   ├── BM1684X
|   │   ├── lprnet_fp32_1b.bmodel                                     # 使用TPU-MLIR编译，用于BM1684X的FP32 LPRNet BModel，batch_size=1，num_core=1
|   │   ├── lprnet_fp16_1b.bmodel                                     # 使用TPU-MLIR编译，用于BM1684X的FP16 LPRNet BModel，batch_size=1，num_core=1
|   │   ├── lprnet_int8_1b.bmodel                                     # 使用TPU-MLIR编译，用于BM1684X的INT8 LPRNet BModel，batch_size=1，num_core=1
|   │   └── lprnet_int8_4b.bmodel                                     # 使用TPU-MLIR编译，用于BM1684X的INT8 LPRNet BModel，batch_size=4，num_core=1
|   ├── BM1688
|   │   ├── lprnet_fp32_1b.bmodel                                     # 使用TPU-MLIR编译，用于BM1688的FP32 LPRNet BModel，batch_size=1，num_core=1
|   │   ├── lprnet_fp16_1b.bmodel                                     # 使用TPU-MLIR编译，用于BM1688的FP16 LPRNet BModel，batch_size=1，num_core=1
|   │   ├── lprnet_int8_1b.bmodel                                     # 使用TPU-MLIR编译，用于BM1688的INT8 LPRNet BModel，batch_size=1，num_core=1
|   │   ├── lprnet_int8_4b.bmodel                                     # 使用TPU-MLIR编译，用于BM1688的INT8 LPRNet BModel，batch_size=4，num_core=1
|   │   ├── lprnet_fp32_1b_2core.bmodel                               # 使用TPU-MLIR编译，用于BM1688的FP32 LPRNet BModel，batch_size=1，num_core=2
|   │   ├── lprnet_fp16_1b_2core.bmodel                               # 使用TPU-MLIR编译，用于BM1688的FP16 LPRNet BModel，batch_size=1，num_core=2
|   │   ├── lprnet_int8_1b_2core.bmodel                               # 使用TPU-MLIR编译，用于BM1688的INT8 LPRNet BModel，batch_size=1，num_core=2
|   │   └── lprnet_int8_4b_2core.bmodel                               # 使用TPU-MLIR编译，用于BM1688的INT8 LPRNet BModel，batch_size=4，num_core=2
│   ├── CV186X
│   │   ├── lprnet_fp16_1b.bmodel                                     # 使用TPU-MLIR编译，用于CV186X的FP16 LPRNet BModel，batch_size=1，num_core=1
│   │   ├── lprnet_fp32_1b.bmodel                                     # 使用TPU-MLIR编译，用于CV186X的FP32 LPRNet BModel，batch_size=1，num_core=1
│   │   ├── lprnet_int8_1b.bmodel                                     # 使用TPU-MLIR编译，用于CV186X的INT8 LPRNet BModel，batch_size=1，num_core=1
│   │   └── lprnet_int8_4b.bmodel                                     # 使用TPU-MLIR编译，用于CV186X的INT8 LPRNet BModel，batch_size=4，num_core=1
|   │── torch
|   │   ├── Final_LPRNet_model.pth                                    # LPRNet 原始模型
|   │   └── LPRNet_model_trace.pt                                     # trace后的JIT LPRNet模型
|   └── onnx
|       ├── lprnet_1b.onnx                                            # 导出的onnx LPRNet模型，batch_size=1
|       └── lprnet_4b.onnx                                            # 导出的onnx LPRNet模型，batch_size=4   
└── yolov5s-licensePLate
    ├── BM1684
    │   ├── yolov5s_v6.1_license_3output_fp32_1b.bmodel               # 用于BM1684的FP32 YOLOv5 BModel，batch_size=1，num_core=1
    │   ├── yolov5s_v6.1_license_3output_int8_1b.bmodel               # 用于BM1684的INT8 YOLOv5 BModel，batch_size=1，num_core=1
    │   └── yolov5s_v6.1_license_3output_int8_4b.bmodel               # 用于BM1684的INT8 YOLOv5 BModel，batch_size=4，num_core=1             
    ├── BM1684X
    │   ├── yolov5s_v6.1_license_3output_fp32_1b.bmodel               # 用于BM1684X的FP32 YOLOv5 BModel，batch_size=1，num_core=1
    │   ├── yolov5s_v6.1_license_3output_int8_1b.bmodel               # 用于BM1684X的INT8 YOLOv5 BModel，batch_size=1，num_core=1
    │   └── yolov5s_v6.1_license_3output_int8_4b.bmodel               # 用于BM1684X的INT8 YOLOv5 BModel，batch_size=4，num_core=1     
    ├── BM1688
    |   ├── yolov5s_v6.1_license_3output_fp32_1b_2core.bmodel         # 用于BM1688的FP32 YOLOv5 BModel，batch_size=1，num_core=2
    |   ├── yolov5s_v6.1_license_3output_fp32_1b.bmodel               # 用于BM1688的FP32 YOLOv5 BModel，batch_size=1，num_core=1
    |   ├── yolov5s_v6.1_license_3output_fp32_4b_2core.bmodel         # 用于BM1688的FP32 YOLOv5 BModel，batch_size=4，num_core=2
    |   ├── yolov5s_v6.1_license_3output_fp32_4b.bmodel               # 用于BM1688的FP32 YOLOv5 BModel，batch_size=4，num_core=1
    |   ├── yolov5s_v6.1_license_3output_int8_1b_2core.bmodel         # 用于BM1688的INT8 YOLOv5 BModel，batch_size=1，num_core=2
    |   ├── yolov5s_v6.1_license_3output_int8_1b.bmodel               # 用于BM1688的INT8 YOLOv5 BModel，batch_size=1，num_core=1
    |   ├── yolov5s_v6.1_license_3output_int8_4b_2core.bmodel         # 用于BM1688的INT8 YOLOv5 BModel，batch_size=4，num_core=2
    |   └── yolov5s_v6.1_license_3output_int8_4b.bmodel               # 用于BM1688的INT8 YOLOv5 BModel，batch_size=4，num_core=1
    └── CV186X
        ├── yolov5s_v6.1_license_3output_fp16_1b.bmodel               # 用于CV186X的FP16 YOLOv5 BModel, batch_size=1, num_core=1
        ├── yolov5s_v6.1_license_3output_fp16_4b.bmodel               # 用于CV186X的FP16 YOLOv5 BModel, batch_size=4, num_core=1
        ├── yolov5s_v6.1_license_3output_fp32_1b.bmodel               # 用于CV186X的FP32 YOLOv5 BModel, batch_size=1, num_core=1
        ├── yolov5s_v6.1_license_3output_fp32_4b.bmodel               # 用于CV186X的FP32 YOLOv5 BModel, batch_size=4, num_core=1
        ├── yolov5s_v6.1_license_3output_int8_1b.bmodel               # 用于CV186X的INT8 YOLOv5 BModel, batch_size=1, num_core=1
        └── yolov5s_v6.1_license_3output_int8_4b.bmodel               # 用于CV186X的INT8 YOLOv5 BModel, batch_size=4, num_core=1

下载的数据包括：
./datasets
├── 1080_1920_30s_512kb.mp4                 # 默认测试视频1
└── 1080_1920_5s.mp4                        # 测试视频2

```

## 4. 模型编译
若需要使用自己的模型，需要保证模型的输入输出和本例程的前后处理相对应。
车牌识别模型编译过程参考[sophon-demo lprnet模型编译](../../sample/LPRNet/README.md#4-模型编译)，注意不能直接用该例程里面的模型。
车牌检测模型编译过程参考[sophon-demo yolov5模型编译](../../sample/YOLOv5/README.md#4-模型编译)，注意不能直接用该例程里面的模型。

> **说明**： 
> 本例程中提供的yolov5s-licenseplate模型为基于绿牌数据集训练的模型，供示例使用参考，无原始模型及精度数据。

## 5. 例程测试
- [C++例程](./cpp/README.md)
- [python例程](./python/README.md)

## 6. 性能测试
### 6.1 bmrt_test
LPRNet性能可参考[LPRNet bmrt_test](../../sample/LPRNet/README.md#71-bmrt_test)里面的性能数据。

使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684/yolov5s_v6.1_license_3output_fp32_1b.bmodel 
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型每张图片的理论推理时间，结果如下：
| 测试模型                                                 | calculate time(ms) |
| -------------------------------------------------------- | ------------------ |
| BM1684/yolov5s_v6.1_license_3output_fp32_1b.bmodel       | 20.6               |
| BM1684/yolov5s_v6.1_license_3output_int8_1b.bmodel       | 12.4               |
| BM1684/yolov5s_v6.1_license_3output_int8_4b.bmodel       | 5.4                |
| BM1684X/yolov5s_v6.1_license_3output_fp32_1b.bmodel      | 18.6               |
| BM1684X/yolov5s_v6.1_license_3output_int8_1b.bmodel      | 2.4                |
| BM1684X/yolov5s_v6.1_license_3output_int8_4b.bmodel      | 1.8                |
| BM1688/yolov5s_v6.1_license_3output_fp32_1b_2core.bmodel | 59.5               |
| BM1688/yolov5s_v6.1_license_3output_fp32_1b.bmodel       | 93.7               |
| BM1688/yolov5s_v6.1_license_3output_fp32_4b_2core.bmodel | 48.0               |
| BM1688/yolov5s_v6.1_license_3output_fp32_4b.bmodel       | 92.9               |
| BM1688/yolov5s_v6.1_license_3output_int8_1b_2core.bmodel | 4.9                |
| BM1688/yolov5s_v6.1_license_3output_int8_1b.bmodel       | 5.8                |
| BM1688/yolov5s_v6.1_license_3output_int8_4b_2core.bmodel | 3.2                |
| BM1688/yolov5s_v6.1_license_3output_int8_4b.bmodel       | 5.6                |
| CV186X/yolov5s_v6.1_license_3output_fp32_1b.bmodel       | 101.8              |
| CV186X/yolov5s_v6.1_license_3output_fp16_1b.bmodel       | 34.9               |
| CV186X/yolov5s_v6.1_license_3output_int8_1b.bmodel       | 12.5               |
| CV186X/yolov5s_v6.1_license_3output_int8_4b.bmodel       | 7.0                |

> **测试说明**：  
> 1. 性能测试结果具有一定的波动性；
> 2. `calculate time`已折算为平均每张图片的推理时间；
> 3. SoC和PCIe的测试结果基本一致。

### 6.2 程序运行性能
参考[例程测试](#5-例程测试)运行程序，并查看统计的total fps。

在不同的测试平台上，使用不同的例程和对应的配置文件，性能测试结果如下：
| 测试平台 | 测试程序      | 测试模型                                                                       | 配置文件           | 路数 | FPS | tpu利用率(%) | cpu利用率(%) | 系统内存占用(MB) | 设备内存占用(MB) |
| -------- | ------------- | ------------------------------------------------------------------------------ | ------------------ | ---- | --- | ------------ | ------------ | ---------------- | ---------------- |
| SE9-8    | vlpr_bmcv.soc | lprnet_int8_4b.bmodel，yolov5s_v6.1_license_3output_int8_4b.bmodel             | config_se9-8.json  | 8    | 89  | 95-100       | 200-250      | 40-50            | 900-1050         |
| SE9-16   | vlpr_bmcv.soc | lprnet_int8_4b_2core.bmodel，yolov5s_v6.1_license_3output_int8_4b_2core.bmodel | config_se9-16.json | 16   | 157 | 80-100       | 410-450      | 50-60            | 3100-3300        |
| SE9-8    | vlpr.py       | lprnet_int8_4b.bmodel，yolov5s_v6.1_license_3output_int8_4b.bmodel             | default            | 8    | 206 | 94-100       | 240-250      | 140-160          | 1720-1740        |
| SE9-16   | vlpr.py       | lprnet_int8_4b_2core.bmodel，yolov5s_v6.1_license_3output_int8_4b_2core.bmodel | default            | 16   | 380 | 95-99        | 400-480      | 272-320          | 3590-3595        |

> **测试说明**：  
> 1. 性能测试结果具有一定的波动性；
> 2. SE9-16的主控处理器为8核CA53@1.6GHz，SE9-8为6核CA53@1.6GHz，PCIe上的性能由于处理器的不同可能存在较大差异； 
> 3. 性能数据在程序启动前和结束前不准确，上面来自程序运行稳定后的数据；
> 4. 各项指标的查看方式可以参考[测试指标查看方式](../../docs/Check_Statis.md)；
> 5. 部署环境下的NPU等设备内存大小会显著影响例程运行的路数。如果默认的输入路数运行中出现了申请内存失败等错误，可以考虑把输入路数减少，或者参考[FAQ](../../docs/FAQ.md#73-程序运行时出现bm_ion_alloc-failed等报错)；
> 6. 若出现申请设备内存失败，错误返回-24，需要把指定同一时间最多可开启的文件数调大一点：比如SoC上可以设置`ulimit -n 4096`。

## 7. FAQ
其他问题请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。
