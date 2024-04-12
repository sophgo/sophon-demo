# C3D
- [1. 简介](#1-简介)
- [2. 特性](#2-特性)
- [3. 准备模型与数据](#3-准备模型与数据)
- [4. 模型编译](#4-模型编译)
- [5. 例程测试](#5-例程测试)
- [6. 精度测试](#6-精度测试)
  - [6.1 测试方法](#61-测试方法)
  - [6.2 测试结果](#62-测试结果)
- [7. 性能测试](#7-性能测试)
  - [7.1 bmrt\_test](#71-bmrt_test)
  - [7.2 程序运行性能](#72-程序运行性能)
- [8. FAQ](#8-faq)



## 1. 简介
C3D是使用三维卷积进行视频动作识别的开荒者，论文链接：[Learning Spatiotemporal Features with 3D Convolutional Networks](https://arxiv.org/abs/1412.0767v4)。

本例程对[MMAction的C3D_UCF101模型](https://mmaction2.readthedocs.io/zh-cn/latest/model_zoo/recognition.html)进行了移植，在相同的预处理流程下可以做到精度对齐。
## 2. 特性
* 支持BM1688/CV186X(SoC)、BM1684X(x86 PCIe、SoC)、BM1684(x86 PCIe、SoC、arm PCIe)
* 支持FP32、FP16(BM1688/BM1684X)、INT8模型编译和推理
* 支持基于BMCV和OpenCV预处理的C++推理
* 支持基于OpenCV预处理的Python推理
* 支持单batch和多batch模型推理
* 支持视频文件夹测试
## 3. 准备模型与数据
建议使用TPU-MLIR编译BModel，Pytorch模型在编译前要导出成onnx模型。

本例程在`scripts`目录下提供了**所有相关的模型和数据集**的下载脚本`download.sh`，您也可以自己准备模型和数据集，并参考[4. 模型转换](#4-模型转换)进行模型转换。

**如果您有自己训练的Pytorch C3D模型**，您可以参考tools/c3d_transform.py，**自行修改源模型路径和模型网络的层名，确保能够加载您的参数**，以成功转换torchscript和onnx模型。同时，您需要准备用于测试的数据集，如果量化模型，还要准备用于量化的数据集。

```bash
# 安装unzip，若已安装请跳过，非ubuntu系统视情况使用yum或其他方式安装
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```
执行后，模型保存在`models`，数据集在`datasets`

下载的模型包括：
```
./models
├── BM1684
│   ├── c3d_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684的FP32 BModel，batch_size=1
│   ├── c3d_fp32_4b.bmodel   # 使用TPU-MLIR编译，用于BM1684的FP32 BModel，batch_size=4
│   ├── c3d_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684的INT8 BModel，batch_size=1
│   └── c3d_int8_4b.bmodel   # 使用TPU-MLIR编译，用于BM1684的INT8 BModel，batch_size=4
├── BM1684X
│   ├── c3d_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=1
│   ├── c3d_fp32_4b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=4
│   ├── c3d_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=1
│   ├── c3d_fp16_4b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=4
│   ├── c3d_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=1
│   └── c3d_int8_4b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=4
├── BM1688
│   ├── c3d_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1，num_core=1
│   ├── c3d_fp32_4b.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=4，num_core=1
│   ├── c3d_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1，num_core=1
│   ├── c3d_fp16_4b.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=4，num_core=1
│   ├── c3d_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=1，num_core=1
│   ├── c3d_int8_4b.bmodel   # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=4，num_core=1
│   ├── c3d_fp32_1b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1，num_core=2
│   ├── c3d_fp32_4b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=4，num_core=2
│   ├── c3d_fp16_1b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1，num_core=2
│   ├── c3d_fp16_4b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=4，num_core=2
│   ├── c3d_int8_1b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=1，num_core=2
│   └── c3d_int8_4b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=4，num_core=2
├── CV186X
│   ├── c3d_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于CV186X的FP32 BModel，batch_size=1
│   ├── c3d_fp32_4b.bmodel   # 使用TPU-MLIR编译，用于CV186X的FP32 BModel，batch_size=4
│   ├── c3d_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于CV186X的FP16 BModel，batch_size=1
│   ├── c3d_fp16_4b.bmodel   # 使用TPU-MLIR编译，用于CV186X的FP16 BModel，batch_size=4
│   ├── c3d_int8_1b.bmodel   # 使用TPU-MLIR编译，用于CV186X的INT8 BModel，batch_size=1
│   └── c3d_int8_4b.bmodel   # 使用TPU-MLIR编译，用于CV186X的INT8 BModel，batch_size=4
│── torch
│   └── c3d_ucf101.pt        # trace后的torchscript模型
└── onnx
    └── c3d_ucf101.onnx      # 导出的onnx动态模型       
```
下载的数据包括：
```
./datasets/UCF_test_01       #UCF101的一个测试子集。
```

## 4. 模型编译

导出的模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。建议使用TPU-MLIR编译BModel。

模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index/material/31/all.html)相应版本的SDK中获取)。

- 生成FP32 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684/BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684 #bm1684x/bm1688/cv186x
```

​执行上述命令会在`models/BM1684`等文件夹下生成`c3d_fp32_1b.bmodel`等文件，即转换好的FP32 BModel。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x #bm1688/cv186x
```

​执行上述命令会在`models/BM1684X/`等文件夹下生成`c3d_fp16_1b.bmodel`等文件，即转换好的FP16 BModel。

- 生成INT8 BModel

​本例程在`scripts`目录下提供了量化INT8 BModel的脚本，请注意修改`gen_int8bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台（**支持BM1684/BM1684X/BM1688/CV186X**），如：

```shell
./scripts/gen_int8bmodel_mlir.sh bm1684 #bm1684x/bm1688/cv186x
```

​上述脚本会在`models/BM1684`等文件夹下生成`c3d_int8_1b.bmodel`等文件，即转换好的INT8 BModel。


**如果您不使用本例程的数据集**，本例程在`tools`目录下提供了准备npy数据的python脚本，用户可以根据脚本自己准备npy格式量化数据集。
```bash
cd tools
python3 c3d_npy.py --input_path ../datasets/UCF_test_01 #for tpu-mlir
```
执行后，会在datasets目录下产生`cali_set_npy`文件夹，可以作为量化模型使用的数据集。

## 5. 例程测试
- [C++例程](./cpp/README.md)
- [Python例程](./python/README.md)

## 6. 精度测试
### 6.1 测试方法

首先，参考[C++例程](cpp/README.md#32-测试视频理解数据集)或[Python例程](python/README.md#22-测试视频理解数据集)推理要测试的数据集，生成预测的json文件。
然后，使用`tools`目录下的`eval_ucf.py`脚本，将测试生成的json文件与测试集标签json文件进行对比，计算出准确率信息，命令如下：
```bash
# 请根据实际情况修改程序路径和json文件路径
python3 tools/eval_ucf.py --gt_path datasets/ground_truth.json --result_json cpp/c3d_bmcv/results/c3d_fp32_1b.bmodel_bmcv_cpp.json
```
### 6.2 测试结果
根据本例程提供的数据集，测试结果如下：
|   测试平台    |      测试程序     |    测试模型        | ACC |
| ------------ | ---------------- | ------------------ | --- |
| SE5-16       | c3d_opencv.py  | c3d_fp32_1b.bmodel     |    0.715 |
| SE5-16       | c3d_opencv.py  | c3d_fp32_4b.bmodel     |    0.715 |
| SE5-16       | c3d_opencv.py  | c3d_int8_1b.bmodel     |    0.712 |
| SE5-16       | c3d_opencv.py  | c3d_int8_4b.bmodel     |    0.712 |
| SE5-16       | c3d_opencv.soc | c3d_fp32_1b.bmodel     |    0.715 |
| SE5-16       | c3d_opencv.soc | c3d_fp32_4b.bmodel     |    0.715 |
| SE5-16       | c3d_opencv.soc | c3d_int8_1b.bmodel     |    0.712 |
| SE5-16       | c3d_opencv.soc | c3d_int8_4b.bmodel     |    0.712 |
| SE5-16       | c3d_bmcv.soc   | c3d_fp32_1b.bmodel     |    0.715 |
| SE5-16       | c3d_bmcv.soc   | c3d_fp32_4b.bmodel     |    0.715 |
| SE5-16       | c3d_bmcv.soc   | c3d_int8_1b.bmodel     |    0.710 |
| SE5-16       | c3d_bmcv.soc   | c3d_int8_4b.bmodel     |    0.710 |
| SE7-32       | c3d_opencv.py  | c3d_fp32_1b.bmodel     |    0.715 |
| SE7-32       | c3d_opencv.py  | c3d_fp32_4b.bmodel     |    0.715 |
| SE7-32       | c3d_opencv.py  | c3d_fp16_1b.bmodel     |    0.715 |
| SE7-32       | c3d_opencv.py  | c3d_fp16_4b.bmodel     |    0.715 |
| SE7-32       | c3d_opencv.py  | c3d_int8_1b.bmodel     |    0.715 |
| SE7-32       | c3d_opencv.py  | c3d_int8_4b.bmodel     |    0.715 |
| SE7-32       | c3d_opencv.soc | c3d_fp32_1b.bmodel     |    0.715 |
| SE7-32       | c3d_opencv.soc | c3d_fp32_4b.bmodel     |    0.715 |
| SE7-32       | c3d_opencv.soc | c3d_fp16_1b.bmodel     |    0.715 |
| SE7-32       | c3d_opencv.soc | c3d_fp16_4b.bmodel     |    0.715 |
| SE7-32       | c3d_opencv.soc | c3d_int8_1b.bmodel     |    0.715 |
| SE7-32       | c3d_opencv.soc | c3d_int8_4b.bmodel     |    0.715 |
| SE7-32       | c3d_bmcv.soc   | c3d_fp32_1b.bmodel     |    0.715 |
| SE7-32       | c3d_bmcv.soc   | c3d_fp32_4b.bmodel     |    0.715 |
| SE7-32       | c3d_bmcv.soc   | c3d_fp16_1b.bmodel     |    0.715 |
| SE7-32       | c3d_bmcv.soc   | c3d_fp16_4b.bmodel     |    0.715 |
| SE7-32       | c3d_bmcv.soc   | c3d_int8_1b.bmodel     |    0.712 |
| SE7-32       | c3d_bmcv.soc   | c3d_int8_4b.bmodel     |    0.712 |
| SE9-16       | c3d_opencv.py  | c3d_fp32_1b.bmodel     |    0.715 |
| SE9-16       | c3d_opencv.py  | c3d_fp32_4b.bmodel     |    0.715 |
| SE9-16       | c3d_opencv.py  | c3d_fp16_1b.bmodel     |    0.715 |
| SE9-16       | c3d_opencv.py  | c3d_fp16_4b.bmodel     |    0.715 |
| SE9-16       | c3d_opencv.py  | c3d_int8_1b.bmodel     |    0.712 |
| SE9-16       | c3d_opencv.py  | c3d_int8_4b.bmodel     |    0.712 |
| SE9-16       | c3d_opencv.soc | c3d_fp32_1b.bmodel     |    0.715 |
| SE9-16       | c3d_opencv.soc | c3d_fp32_4b.bmodel     |    0.715 |
| SE9-16       | c3d_opencv.soc | c3d_fp16_1b.bmodel     |    0.715 |
| SE9-16       | c3d_opencv.soc | c3d_fp16_4b.bmodel     |    0.715 |
| SE9-16       | c3d_opencv.soc | c3d_int8_1b.bmodel     |    0.712 |
| SE9-16       | c3d_opencv.soc | c3d_int8_4b.bmodel     |    0.712 |
| SE9-16       | c3d_bmcv.soc   | c3d_fp32_1b.bmodel     |    0.715 |
| SE9-16       | c3d_bmcv.soc   | c3d_fp32_4b.bmodel     |    0.715 |
| SE9-16       | c3d_bmcv.soc   | c3d_fp16_1b.bmodel     |    0.715 |
| SE9-16       | c3d_bmcv.soc   | c3d_fp16_4b.bmodel     |    0.715 |
| SE9-16       | c3d_bmcv.soc   | c3d_int8_1b.bmodel     |    0.715 |
| SE9-16       | c3d_bmcv.soc   | c3d_int8_4b.bmodel     |    0.715 |
| SE9-16       | c3d_opencv.py  | c3d_fp32_1b_2core.bmodel |    0.715 |
| SE9-16       | c3d_opencv.py  | c3d_fp32_4b_2core.bmodel |    0.715 |
| SE9-16       | c3d_opencv.py  | c3d_fp16_1b_2core.bmodel |    0.715 |
| SE9-16       | c3d_opencv.py  | c3d_fp16_4b_2core.bmodel |    0.715 |
| SE9-16       | c3d_opencv.py  | c3d_int8_1b_2core.bmodel |    0.712 |
| SE9-16       | c3d_opencv.py  | c3d_int8_4b_2core.bmodel |    0.712 |
| SE9-16       | c3d_opencv.soc | c3d_fp32_1b_2core.bmodel |    0.715 |
| SE9-16       | c3d_opencv.soc | c3d_fp32_4b_2core.bmodel |    0.715 |
| SE9-16       | c3d_opencv.soc | c3d_fp16_1b_2core.bmodel |    0.715 |
| SE9-16       | c3d_opencv.soc | c3d_fp16_4b_2core.bmodel |    0.715 |
| SE9-16       | c3d_opencv.soc | c3d_int8_1b_2core.bmodel |    0.712 |
| SE9-16       | c3d_opencv.soc | c3d_int8_4b_2core.bmodel |    0.712 |
| SE9-16       | c3d_bmcv.soc   | c3d_fp32_1b_2core.bmodel |    0.715 |
| SE9-16       | c3d_bmcv.soc   | c3d_fp32_4b_2core.bmodel |    0.715 |
| SE9-16       | c3d_bmcv.soc   | c3d_fp16_1b_2core.bmodel |    0.715 |
| SE9-16       | c3d_bmcv.soc   | c3d_fp16_4b_2core.bmodel |    0.715 |
| SE9-16       | c3d_bmcv.soc   | c3d_int8_1b_2core.bmodel |    0.715 |
| SE9-16       | c3d_bmcv.soc   | c3d_int8_4b_2core.bmodel |    0.715 |
| SE9-8        | c3d_opencv.py  | c3d_fp32_1b.bmodel     |    0.715 |
| SE9-8        | c3d_opencv.py  | c3d_fp32_4b.bmodel     |    0.715 |
| SE9-8        | c3d_opencv.py  | c3d_fp16_1b.bmodel     |    0.715 |
| SE9-8        | c3d_opencv.py  | c3d_fp16_4b.bmodel     |    0.715 |
| SE9-8        | c3d_opencv.py  | c3d_int8_1b.bmodel     |    0.712 |
| SE9-8        | c3d_opencv.py  | c3d_int8_4b.bmodel     |    0.712 |
| SE9-8        | c3d_opencv.soc | c3d_fp32_1b.bmodel     |    0.715 |
| SE9-8        | c3d_opencv.soc | c3d_fp32_4b.bmodel     |    0.715 |
| SE9-8        | c3d_opencv.soc | c3d_fp16_1b.bmodel     |    0.715 |
| SE9-8        | c3d_opencv.soc | c3d_fp16_4b.bmodel     |    0.715 |
| SE9-8        | c3d_opencv.soc | c3d_int8_1b.bmodel     |    0.712 |
| SE9-8        | c3d_opencv.soc | c3d_int8_4b.bmodel     |    0.712 |
| SE9-8        | c3d_bmcv.soc   | c3d_fp32_1b.bmodel     |    0.715 |
| SE9-8        | c3d_bmcv.soc   | c3d_fp32_4b.bmodel     |    0.715 |
| SE9-8        | c3d_bmcv.soc   | c3d_fp16_1b.bmodel     |    0.715 |
| SE9-8        | c3d_bmcv.soc   | c3d_fp16_4b.bmodel     |    0.715 |
| SE9-8        | c3d_bmcv.soc   | c3d_int8_1b.bmodel     |    0.715 |
| SE9-8        | c3d_bmcv.soc   | c3d_int8_4b.bmodel     |    0.715 |

> **测试说明**：  
> 1. 由于sdk版本之间可能存在差异，实际运行结果与本表有<0.01的精度误差是正常的；
> 2. 在搭载了相同TPU和SOPHONSDK的PCIe或SoC平台上，相同程序的精度一致，SE5系列对应BM1684，SE7系列对应BM1684X，SE9系列中SE9-16对应BM1688，SE9-8对应CV186X；

## 7. 性能测试
### 7.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684/c3d_fp32_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是理论推理时间。
测试各个模型的理论推理时间，结果如下：

| 测试模型                       | calculate time(ms) |
| -------------------                |  -------------- |
| BM1684/c3d_fp32_1b.bmodel          |          62.37  |
| BM1684/c3d_fp32_4b.bmodel          |          50.10  |
| BM1684/c3d_int8_1b.bmodel          |          28.25  |
| BM1684/c3d_int8_4b.bmodel          |           7.39  |
| BM1684X/c3d_fp32_1b.bmodel         |          79.05  |
| BM1684X/c3d_fp32_4b.bmodel         |          73.64  |
| BM1684X/c3d_fp16_1b.bmodel         |           9.50  |
| BM1684X/c3d_fp16_4b.bmodel         |           7.11  |
| BM1684X/c3d_int8_1b.bmodel         |           5.57  |
| BM1684X/c3d_int8_4b.bmodel         |           4.41  |
| BM1688/c3d_fp32_1b.bmodel          |         414.33  |
| BM1688/c3d_fp32_4b.bmodel          |         396.96  |
| BM1688/c3d_fp16_1b.bmodel          |          74.94  |
| BM1688/c3d_fp16_4b.bmodel          |          68.31  |
| BM1688/c3d_int8_1b.bmodel          |          34.74  |
| BM1688/c3d_int8_4b.bmodel          |          31.43  |
| BM1688/c3d_fp32_1b_2core.bmodel    |         413.43  |
| BM1688/c3d_fp32_4b_2core.bmodel    |         397.42  |
| BM1688/c3d_fp16_1b_2core.bmodel    |          61.02  |
| BM1688/c3d_fp16_4b_2core.bmodel    |          54.17  |
| BM1688/c3d_int8_1b_2core.bmodel    |          31.54  |
| BM1688/c3d_int8_4b_2core.bmodel    |          28.24  |
| CV186X/c3d_fp32_1b.bmodel          |         417.85  |
| CV186X/c3d_fp32_4b.bmodel          |         394.11  |
| CV186X/c3d_fp16_1b.bmodel          |          76.09  |
| CV186X/c3d_fp16_4b.bmodel          |          65.99  |
| CV186X/c3d_int8_1b.bmodel          |          32.57  |
| CV186X/c3d_int8_4b.bmodel          |          27.78  |

> **测试说明**：  
1. 性能测试结果具有一定的波动性；
2. `calculate time`已折算为每个视频平均推理时间；
3. SoC和PCIe的测试结果基本一致。

### 7.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的视频解码时间、预处理时间、推理时间、后处理时间。C++和Python例程打印的时间已经折算为单张图片的处理时间。

在不同的测试平台上，使用不同的例程、模型测试`datasets/UCF_test_01`，性能测试结果如下：
|    测试平台  |     测试程序  |      测试模型     |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ------------- | ---------------- | -------- | ---------   | ---------------| --------- |
|   SE5-16    |   c3d_opencv.py   |   c3d_fp32_1b.bmodel    |      66.43      |      30.22      |      68.69      |      0.09       |
|   SE5-16    |   c3d_opencv.py   |   c3d_fp32_4b.bmodel    |      67.00      |      37.55      |      56.48      |      0.03       |
|   SE5-16    |   c3d_opencv.py   |   c3d_int8_1b.bmodel    |      66.39      |      30.39      |      34.65      |      0.09       |
|   SE5-16    |   c3d_opencv.py   |   c3d_int8_4b.bmodel    |      67.18      |      37.69      |      13.71      |      0.03       |
|   SE5-16    |  c3d_opencv.soc   |   c3d_fp32_1b.bmodel    |      71.78      |      26.17      |      62.32      |      0.01       |
|   SE5-16    |  c3d_opencv.soc   |   c3d_fp32_4b.bmodel    |      71.94      |      25.91      |      50.09      |      0.00       |
|   SE5-16    |  c3d_opencv.soc   |   c3d_int8_1b.bmodel    |      71.73      |      26.07      |      28.22      |      0.01       |
|   SE5-16    |  c3d_opencv.soc   |   c3d_int8_4b.bmodel    |      71.67      |      25.80      |      7.39       |      0.00       |
|   SE5-16    |   c3d_bmcv.soc    |   c3d_fp32_1b.bmodel    |      75.55      |      6.74       |      62.29      |      0.01       |
|   SE5-16    |   c3d_bmcv.soc    |   c3d_fp32_4b.bmodel    |      74.63      |      6.62       |      50.08      |      0.00       |
|   SE5-16    |   c3d_bmcv.soc    |   c3d_int8_1b.bmodel    |      74.72      |      6.72       |      28.21      |      0.01       |
|   SE5-16    |   c3d_bmcv.soc    |   c3d_int8_4b.bmodel    |      74.97      |      6.57       |      7.38       |      0.00       |
|   SE7-32    |   c3d_opencv.py   |   c3d_fp32_1b.bmodel    |      65.80      |      30.95      |      86.39      |      0.09       |
|   SE7-32    |   c3d_opencv.py   |   c3d_fp32_4b.bmodel    |      67.21      |      38.68      |      80.74      |      0.03       |
|   SE7-32    |   c3d_opencv.py   |   c3d_fp16_1b.bmodel    |      66.06      |      31.05      |      16.78      |      0.09       |
|   SE7-32    |   c3d_opencv.py   |   c3d_fp16_4b.bmodel    |      67.21      |      38.67      |      14.18      |      0.03       |
|   SE7-32    |   c3d_opencv.py   |   c3d_int8_1b.bmodel    |      65.83      |      30.88      |      12.88      |      0.09       |
|   SE7-32    |   c3d_opencv.py   |   c3d_int8_4b.bmodel    |      67.07      |      38.60      |      11.53      |      0.03       |
|   SE7-32    |  c3d_opencv.soc   |   c3d_fp32_1b.bmodel    |      71.89      |      26.43      |      79.06      |      0.01       |
|   SE7-32    |  c3d_opencv.soc   |   c3d_fp32_4b.bmodel    |      72.32      |      26.08      |      73.65      |      0.00       |
|   SE7-32    |  c3d_opencv.soc   |   c3d_fp16_1b.bmodel    |      71.86      |      26.39      |      9.48       |      0.01       |
|   SE7-32    |  c3d_opencv.soc   |   c3d_fp16_4b.bmodel    |      72.34      |      26.15      |      7.11       |      0.00       |
|   SE7-32    |  c3d_opencv.soc   |   c3d_int8_1b.bmodel    |      72.16      |      26.40      |      5.57       |      0.01       |
|   SE7-32    |  c3d_opencv.soc   |   c3d_int8_4b.bmodel    |      72.36      |      26.14      |      4.40       |      0.01       |
|   SE7-32    |   c3d_bmcv.soc    |   c3d_fp32_1b.bmodel    |      74.45      |      3.64       |      79.03      |      0.01       |
|   SE7-32    |   c3d_bmcv.soc    |   c3d_fp32_4b.bmodel    |      75.02      |      3.48       |      73.63      |      0.00       |
|   SE7-32    |   c3d_bmcv.soc    |   c3d_fp16_1b.bmodel    |      74.71      |      3.60       |      9.46       |      0.01       |
|   SE7-32    |   c3d_bmcv.soc    |   c3d_fp16_4b.bmodel    |      74.66      |      3.49       |      7.10       |      0.00       |
|   SE7-32    |   c3d_bmcv.soc    |   c3d_int8_1b.bmodel    |      74.84      |      3.62       |      5.52       |      0.01       |
|   SE7-32    |   c3d_bmcv.soc    |   c3d_int8_4b.bmodel    |      75.16      |      3.49       |      4.41       |      0.00       |
|   SE9-16    |   c3d_opencv.py   |   c3d_fp32_1b.bmodel    |      91.91      |      42.36      |     414.42      |      0.13       |
|   SE9-16    |   c3d_opencv.py   |   c3d_fp32_4b.bmodel    |      94.88      |      50.27      |     397.28      |      0.05       |
|   SE9-16    |   c3d_opencv.py   |   c3d_fp16_1b.bmodel    |      90.77      |      42.20      |      78.45      |      0.13       |
|   SE9-16    |   c3d_opencv.py   |   c3d_fp16_4b.bmodel    |      94.31      |      50.30      |      72.12      |      0.04       |
|   SE9-16    |   c3d_opencv.py   |   c3d_int8_1b.bmodel    |      91.94      |      42.05      |      34.79      |      0.13       |
|   SE9-16    |   c3d_opencv.py   |   c3d_int8_4b.bmodel    |      93.83      |      50.44      |      31.97      |      0.04       |
|   SE9-16    |  c3d_opencv.soc   |   c3d_fp32_1b.bmodel    |     132.62      |     387.37      |     405.13      |      0.02       |
|   SE9-16    |  c3d_opencv.soc   |   c3d_fp32_4b.bmodel    |     132.72      |     387.10      |     388.00      |      0.01       |
|   SE9-16    |  c3d_opencv.soc   |   c3d_fp16_1b.bmodel    |     132.28      |     387.39      |      69.19      |      0.02       |
|   SE9-16    |  c3d_opencv.soc   |   c3d_fp16_4b.bmodel    |     132.06      |     387.35      |      62.73      |      0.01       |
|   SE9-16    |  c3d_opencv.soc   |   c3d_int8_1b.bmodel    |     131.51      |     387.38      |      25.53      |      0.02       |
|   SE9-16    |  c3d_opencv.soc   |   c3d_int8_4b.bmodel    |     132.48      |     387.16      |      22.49      |      0.01       |
|   SE9-16    |   c3d_bmcv.soc    |   c3d_fp32_1b.bmodel    |     142.28      |      11.06      |     405.10      |      0.02       |
|   SE9-16    |   c3d_bmcv.soc    |   c3d_fp32_4b.bmodel    |     142.33      |      10.91      |     387.99      |      0.01       |
|   SE9-16    |   c3d_bmcv.soc    |   c3d_fp16_1b.bmodel    |     143.40      |      11.16      |      69.17      |      0.02       |
|   SE9-16    |   c3d_bmcv.soc    |   c3d_fp16_4b.bmodel    |     142.57      |      11.13      |      62.73      |      0.01       |
|   SE9-16    |   c3d_bmcv.soc    |   c3d_int8_1b.bmodel    |     143.75      |      11.41      |      25.50      |      0.02       |
|   SE9-16    |   c3d_bmcv.soc    |   c3d_int8_4b.bmodel    |     142.03      |      10.91      |      22.48      |      0.01       |
|   SE9-16    |   c3d_opencv.py   |c3d_fp32_1b_2core.bmodel |      92.42      |      42.31      |     413.57      |      0.13       |
|   SE9-16    |   c3d_opencv.py   |c3d_fp32_4b_2core.bmodel |      94.70      |      50.44      |     397.63      |      0.05       |
|   SE9-16    |   c3d_opencv.py   |c3d_fp16_1b_2core.bmodel |      92.57      |      41.98      |      64.42      |      0.13       |
|   SE9-16    |   c3d_opencv.py   |c3d_fp16_4b_2core.bmodel |      94.33      |      50.32      |      58.00      |      0.04       |
|   SE9-16    |   c3d_opencv.py   |c3d_int8_1b_2core.bmodel |      92.22      |      42.16      |      31.64      |      0.13       |
|   SE9-16    |   c3d_opencv.py   |c3d_int8_4b_2core.bmodel |      94.32      |      50.60      |      29.26      |      0.04       |
|   SE9-16    |  c3d_opencv.soc   |c3d_fp32_1b_2core.bmodel |     132.71      |     387.49      |     404.24      |      0.02       |
|   SE9-16    |  c3d_opencv.soc   |c3d_fp32_4b_2core.bmodel |     133.12      |     387.29      |     388.47      |      0.01       |
|   SE9-16    |  c3d_opencv.soc   |c3d_fp16_1b_2core.bmodel |     132.91      |     387.52      |      55.13      |      0.02       |
|   SE9-16    |  c3d_opencv.soc   |c3d_fp16_4b_2core.bmodel |     133.25      |     387.12      |      48.66      |      0.01       |
|   SE9-16    |  c3d_opencv.soc   |c3d_int8_1b_2core.bmodel |     132.66      |     387.34      |      22.35      |      0.02       |
|   SE9-16    |  c3d_opencv.soc   |c3d_int8_4b_2core.bmodel |     132.56      |     387.10      |      19.32      |      0.01       |
|   SE9-16    |   c3d_bmcv.soc    |c3d_fp32_1b_2core.bmodel |     143.02      |      11.32      |     404.23      |      0.02       |
|   SE9-16    |   c3d_bmcv.soc    |c3d_fp32_4b_2core.bmodel |     142.85      |      10.80      |     388.46      |      0.01       |
|   SE9-16    |   c3d_bmcv.soc    |c3d_fp16_1b_2core.bmodel |     142.27      |      11.26      |      55.10      |      0.02       |
|   SE9-16    |   c3d_bmcv.soc    |c3d_fp16_4b_2core.bmodel |     142.10      |      10.93      |      48.67      |      0.01       |
|   SE9-16    |   c3d_bmcv.soc    |c3d_int8_1b_2core.bmodel |     142.53      |      11.21      |      22.31      |      0.02       |
|   SE9-16    |   c3d_bmcv.soc    |c3d_int8_4b_2core.bmodel |     142.31      |      10.81      |      19.31      |      0.01       |
|    SE9-8    |   c3d_opencv.py   |   c3d_fp32_1b.bmodel    |      91.67      |      41.95      |     427.43      |      0.13       |
|    SE9-8    |   c3d_opencv.py   |   c3d_fp32_4b.bmodel    |      93.11      |      50.04      |     403.67      |      0.05       |
|    SE9-8    |   c3d_opencv.py   |   c3d_fp16_1b.bmodel    |      91.76      |      42.03      |      85.14      |      0.13       |
|    SE9-8    |   c3d_opencv.py   |   c3d_fp16_4b.bmodel    |      88.07      |      49.75      |      75.09      |      0.05       |
|    SE9-8    |   c3d_opencv.py   |   c3d_int8_1b.bmodel    |      86.82      |      42.07      |      41.89      |      0.13       |
|    SE9-8    |   c3d_opencv.py   |   c3d_int8_4b.bmodel    |      87.39      |      49.91      |      37.04      |      0.04       |
|    SE9-8    |  c3d_opencv.soc   |   c3d_fp32_1b.bmodel    |     120.19      |      33.83      |     418.01      |      0.02       |
|    SE9-8    |  c3d_opencv.soc   |   c3d_fp32_4b.bmodel    |     119.16      |      33.51      |     394.04      |      0.01       |
|    SE9-8    |  c3d_opencv.soc   |   c3d_fp16_1b.bmodel    |     119.90      |      33.60      |      75.82      |      0.02       |
|    SE9-8    |  c3d_opencv.soc   |   c3d_fp16_4b.bmodel    |     118.36      |      33.43      |      66.01      |      0.01       |
|    SE9-8    |  c3d_opencv.soc   |   c3d_int8_1b.bmodel    |     119.01      |      33.79      |      32.55      |      0.02       |
|    SE9-8    |  c3d_opencv.soc   |   c3d_int8_4b.bmodel    |     118.10      |      33.40      |      27.83      |      0.01       |
|    SE9-8    |   c3d_bmcv.soc    |   c3d_fp32_1b.bmodel    |     130.30      |      9.05       |     418.00      |      0.02       |
|    SE9-8    |   c3d_bmcv.soc    |   c3d_fp32_4b.bmodel    |     130.70      |      8.84       |     394.03      |      0.01       |
|    SE9-8    |   c3d_bmcv.soc    |   c3d_fp16_1b.bmodel    |     131.08      |      9.00       |      75.79      |      0.02       |
|    SE9-8    |   c3d_bmcv.soc    |   c3d_fp16_4b.bmodel    |     128.82      |      8.75       |      66.03      |      0.01       |
|    SE9-8    |   c3d_bmcv.soc    |   c3d_int8_1b.bmodel    |     130.24      |      8.95       |      32.53      |      0.02       |
|    SE9-8    |   c3d_bmcv.soc    |   c3d_int8_4b.bmodel    |     128.00      |      8.81       |      27.79      |      0.01       |

> **测试说明**：  
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. SE5-16/SE7-32的主控处理器均为8核CA53@2.3GHz，SE9-16的主控处理器为8核CA53@1.6GHz，SE9-8为6核CA53@1.6GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异，不同的阈值对后处理时间影响较大。
> 5. C3D的后处理只有argmax，耗时很短，可以忽略。

## 8. FAQ
请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。