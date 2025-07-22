# D-FINE

## 目录

- [D-FINE](#dfine)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
    - [2.1 目录结构说明](#21-目录结构说明)
    - [2.2 SDK特性](#22-sdk特性)
  - [3. 数据准备与模型编译](#3-数据准备与模型编译)
    - [3.1 数据准备](#31-数据准备)
    - [3.2 模型编译](#32-模型编译)
  - [4. 例程测试](#4-例程测试)
  - [5. 精度测试](#5-精度测试)
    - [5.1 测试方法](#51-测试方法)
    - [5.2 测试结果](#52-测试结果)
  - [6. 性能测试](#6-性能测试)
    - [6.1 程序运行性能](#62-程序运行性能)
  - [7. FAQ](#7-faq)
  
## 1. 简介
D-FINE 是一个强大的实时目标检测器，将 DETR 中的边界框回归任务重新定义为了细粒度的分布优化（FDR），并引入全局最优的定位自蒸馏（GO-LSD），在不增加额外推理和训练成本的情况下，实现了卓越的性能。目前已适配[D-FINE官方开源仓库](https://github.com/Peterande/D-FINE?tab=readme-ov-file)，支持在SOPHON BM1684X/BM1688/CV186X上进行推理测试。

## 2. 特性

### 2.1 目录结构说明
```bash
├── python                # 存放Python例程及其README
|   ├──README.md 
|   ├──dfine_bmcv.py     # Python例程
|   └──...                # Python例程共用功能的封装。
├── README.md             # 本例程的中文指南
└── scripts               # 存放模型编译、数据下载等shell脚本
```

### 2.2 SDK特性
* 支持BM1688/CV186X(SoC)和BM1684X(x86 PCIe、SoC、riscv PCIe)
* 支持FP32、FP16(BM1684X/BM1688/CV186X)
* 支持Python推理
* 支持图片和视频测试

## 3. 数据准备与模型编译

### 3.1 数据准备

​本例程在`scripts`目录下提供了相关模型和数据的下载脚本`download.sh`，**如果您希望自己准备模型和数据集，可以跳过本小节，参考[3.2 模型编译](#32-模型编译)进行模型转换。**

```bash
chmod -R +x scripts/
./scripts/download.sh
```

下载的模型包括：
```bash
models/
├── BM1684X # 在BM1684X上运行的模型
│   ├── dfine_n_coco_f16_1b.bmodel
│   └── dfine_s_obj2coco_f16_1b.bmodel
├── BM1688 # 在BM1688上运行的模型
│   ├── dfine_n_coco_f16_1b.bmodel
│   ├── dfine_n_coco_f16_2core.bmodel
│   ├── dfine_s_obj2coco_f16_1b.bmodel
│   └── dfine_s_obj2coco_f16_1b_2core.bmodel
├── onnx
    ├── dfine_n_coco.onnx
    └── dfine_s_obj2coco.onnx
```
下载的数据包括：
```bash
./datasets
├── test                                      # 测试图片
├── test_car_person_1080P.mp4                 # 测试视频
├── coco.names                                # coco类别名文件
└── coco128                                   # coco128数据集，用于模型量化                                  
```

### 3.2 模型编译

**如果您不编译模型，只想直接使用下载的数据集和模型，可以跳过本小节。**

源模型需要编译成BModel才能在SOPHON TPU上运行，源模型在编译前要导出成onnx模型，如果您使用的TPU-MLIR版本>=v1.3.0（即官网v23.07.01），也可以直接使用torchscript模型。具体可参考[官方onnx模型导出](https://github.com/Peterande/D-FINE/tree/master?tab=readme-ov-file#tools)。​同时，您需要准备用于测试的数据集，如果量化模型，还要准备用于量化的数据集。

建议使用TPU-MLIR编译BModel，模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录，并使用本例程提供的脚本将onnx模型编译为BModel。脚本中命令的详细说明可参考《TPU-MLIR开发手册》(请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的SDK中获取)。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688**），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x #bm1688/cv186x
```

​执行上述命令会在`models/BM1684X/`等文件夹下生成转换好的FP16 BModel。

注：这里用到了混合精度量化，需要将一些层设为敏感层，相应的qtable在此前`download.sh`下载的`models/onnx`文件夹里。如果您需要量化自己微调过的模型，可以参考[量化指南](../../docs/Calibration_Guide.md#13-特定模型优化技巧)中的方法，从我们提供的qtable倒推出自己模型需要的qtable。BM1684不支持F16混合精度，如果您使用BM1684系列产品，您需要把qtable中的F16层更改为F32。

## 4. 例程测试
- [Python例程](./python/README.md)

## 5. 精度测试
### 5.1 测试方法

首先，参考[C++例程](cpp/README.md#32-测试图片)或[Python例程](python/README.md#22-测试图片)推理要测试的数据集，生成预测的json文件，注意修改数据集(datasets/coco/val2017_1000)，并将self.postprocess中的参数 thrh 调整为 0.25。

然后，使用`tools`目录下的`eval_coco.py`脚本，将测试生成的json文件与测试集标签json文件进行对比，计算出目标检测的评价指标，命令如下：
```bash
# 安装pycocotools，若已安装请跳过
pip3 install pycocotools==2.0.8
# 请根据实际情况修改程序路径和json文件路径
python3 tools/eval_coco.py --gt_path datasets/coco/instances_val2017_1000.json --result_json results/dfine_s_obj2coco_f16_1b.bmodel_val2017_1000_bmcv_python_result.json
```
### 5.2 测试结果
在coco2017 val数据集上，精度测试结果如下：
|   测试平台    |      测试程序     |      测试模型          |AP@IoU=0.5:0.95|AP@IoU=0.5|
| ------------ | ---------------- | ---------------------- | ------------- | -------- |
|   SE7-32    |  dfine_bmcv.py   |      dfine_n_coco_f16_1b.bmodel       | 0.381 | 0.538 |
|   SE7-32    |  dfine_bmcv.py  |      dfine_s_obj2coco_f16_1b.bmodel       | 0.450 | 0.616 |

> **测试说明**：  
> 1. 本次仅在SE7系列平台上进行了测试，SE9-16和SE9-8平台上运行相同模型和程序时，精度表现与SE7-32平台基本一致，实际运行结果与本表有<0.01的精度误差是正常的；
> 2. AP@IoU=0.5:0.95为area=all对应的指标；
> 3. 在搭载了相同TPU和SOPHONSDK的PCIe或SoC平台上，相同程序的精度一致，SE5系列对应BM1684，SE7系列对应BM1684X，SE9系列中，SE9-16对应BM1688，SE9-8对应CV186X；

## 6. 性能测试
### 6.1 程序运行性能
参考[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++和Python例程打印的时间已经折算为单张图片的处理时间。

在不同的测试平台上，使用不同的例程、模型测试`datasets/coco/val2017_1000`，性能测试结果如下：
|    测试平台  |     测试程序      |        测试模型        |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ---------------- | ---------------------- | -------- | --------- | --------- | --------- |
|   SE7-32    |  dfine_bmcv.py   |      dfine_n_coco_f16_1b.bmodel       |      3.48       |      2.34       |      9.28      |      0.24       |
|   SE7-32    |  dfine_bmcv.py   |      dfine_s_obj2coco_f16_1b.bmodel       |      3.51       |      2.36       |      14.11      |      0.26       |
|    SE9-16    |  dfine_bmcv.py   |      dfine_n_coco_f16_1b.bmodel       |      3.87       |      3.99       |     19.50      |      0.32       |
|    SE9-16    |  dfine_bmcv.py   |      dfine_n_coco_f16_1b_2core.bmodel       |      3.89       |      3.99       |     16.07      |      0.32       |
|    SE9-16    |  dfine_bmcv.py   |      dfine_s_obj2coco_f16_1b.bmodel       |      3.81       |      4.00       |     40.85      |      0.34       |
|    SE9-16    |  dfine_bmcv.py   |      dfine_s_obj2coco_f16_1b_2core.bmodel       |      3.93       |      3.99       |     29.53      |      0.34       |
|    SE9-8    |  dfine_bmcv.py   |      dfine_n_coco_f16_1b.bmodel       |      5.08       |      4.54       |     19.94      |      0.34       |
|    SE9-8    |  dfine_bmcv.py   |      dfine_s_obj2coco_f16_1b.bmodel       |      4.83       |      4.51       |     41.26      |      0.36       |

> **测试说明**：  
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. SE5-16/SE7-32的主控处理器均为8核CA53@2.3GHz，SE9-16为8核CA53@1.6GHz，SE9-8为6核CA53@1.6GHzPCIe上的性能由于处理器的不同可能存在较大差异；
> 4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异，不同的阈值对后处理时间影响较大。 


## 7. FAQ
请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。
