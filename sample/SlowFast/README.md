# SlowFast
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
SlowFast 是 Facebook AI Research (FAIR) 提出的用于视频理解的深度学习模型，特别擅长处理涉及时序动态的任务，比如视频行为识别，论文链接：[SlowFast Networks for Video Recognition](https://arxiv.org/abs/1812.03982)。

本例程对[pytorchvideo的SlowFast R50模型](https://github.com/facebookresearch/pytorchvideo)进行了移植，在相同的预处理流程下可以做到精度对齐。
## 2. 特性
* 支持BM1688(SoC)、BM1684X(PCIe、SoC)
* 支持FP32、FP16、INT8模型编译和推理
* 支持基于OpenCV预处理的C++推理
* 支持基于OpenCV预处理的Python推理
* 支持单batch和多batch模型推理
* 支持视频文件夹测试

## 3. 准备模型与数据
建议使用TPU-MLIR编译BModel，Pytorch模型在编译前要导出成onnx模型。
在官方demo[torchhub_inference_tutorial.ipynb](https://github.com/facebookresearch/pytorchvideo/blob/main/tutorials/torchhub_inference_tutorial.ipynb)的基础上，执行以下部分即可转出onnx模型。
```python
with torch.no_grad():
    torch.onnx.export(model,
            inputs,
            "slowfast_r50.onnx",
            opset_version=13,
            input_names=["input_slow","input_fast"],
            output_names=["output"],
            dynamic_axes={"input_slow":{0:"batch_size"},
                          "input_fast":{0:"batch_size"},
            "output":{0:"batch_size"}})
```

本例程在`scripts`目录下提供了**所有相关的模型和数据集**的下载脚本`download.sh`，您也可以自己准备模型和数据集，并参考[4. 模型转换](#4-模型转换)进行模型转换。

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
├── BM1684X
│   ├── slowfast_bm1684x_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=1
│   ├── slowfast_bm1684x_fp32_4b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=4
│   ├── slowfast_bm1684x_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=1
│   ├── slowfast_bm1684x_fp16_4b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=4
│   ├── slowfast_bm1684x_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=1
│   └── slowfast_bm1684x_int8_4b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=4
├── BM1688
│   ├── slowfast_bm1688_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1，num_core=1
│   ├── slowfast_bm1688_fp32_4b.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=4，num_core=1
│   ├── slowfast_bm1688_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1，num_core=1
│   ├── slowfast_bm1688_fp16_4b.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=4，num_core=1
│   ├── slowfast_bm1688_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=1，num_core=1
│   ├── slowfast_bm1688_int8_4b.bmodel   # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=4，num_core=1
│   ├── slowfast_bm1688_fp32_1b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1，num_core=2
│   ├── slowfast_bm1688_fp32_4b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=4，num_core=2
│   ├── slowfast_bm1688_fp16_1b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1，num_core=2
│   ├── slowfast_bm1688_fp16_4b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=4，num_core=2
│   ├── slowfast_bm1688_int8_1b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=1，num_core=2
│   └── slowfast_bm1688_int8_4b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=4，num_core=2
└── onnx
    └── slowfast_r50.onnx      # 导出的onnx模型
```
下载的数据包括：
```
./datasets/sampled_k400       #Kinetics400的一个测试子集。
```

## 4. 模型编译

导出的模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。建议使用TPU-MLIR编译BModel。

模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的SDK中获取)。

- 生成FP32 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688**），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684x #bm1684x/bm1688
```

​执行上述命令会在`models/BM1684X`等文件夹下生成`slowfast_bm1684x_fp32_1b.bmodel`等文件，即转换好的FP32 BModel。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688**），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x #bm1684x/bm1688
```

​执行上述命令会在`models/BM1684X/`等文件夹下生成`slowfast_bm1684x_fp16_1b.bmodel`等文件，即转换好的FP16 BModel。

- 生成INT8 BModel

​本例程在`scripts`目录下提供了量化INT8 BModel的脚本，请注意修改`gen_int8bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台（**支持BM1684X/BM1688**），如：

```shell
./scripts/gen_int8bmodel_mlir.sh bm1684x #bm1684x/bm1688
```

​上述脚本会在`models/BM1684x`等文件夹下生成`slowfast_bm1684x_int8_1b.bmodel`等文件，即转换好的INT8 BModel。


**如果您不使用本例程的数据集**，本例程在`tools`目录下提供了准备npy数据的python脚本，用户可以根据脚本自己准备npy格式量化数据集。
```bash
cd tools
python3 slowfast_npy.py --input_path ../datasets/sampled_k400 #for tpu-mlir
```
执行后，会在datasets目录下产生`cali_set_npy`文件夹，可以作为量化模型使用的数据集。

## 5. 例程测试
- [C++例程](./cpp/README.md)
- [Python例程](./python/README.md)

## 6. 精度测试
### 6.1 测试方法

首先，参考[C++例程](cpp/README.md#32-测试视频理解数据集)或[Python例程](python/README.md#22-测试视频理解数据集)推理要测试的数据集，生成预测的json文件。
然后，使用`tools`目录下的`eval_kinetics.py`脚本，将测试生成的json文件与测试集标签json文件进行对比，计算出准确率信息，命令如下：
```bash
# 请根据实际情况修改程序路径和json文件路径
python3 tools/eval_kinetics.py --gt_path datasets/ground_truth.json --result_json cpp/slowfast_opencv/results/slowafst_bm1684x_fp32_1b.bmodel_opencv_cpp.json
```
### 6.2 测试结果
根据本例程提供的数据集，测试结果如下：
|   测试平台    |      测试程序     |    测试模型        | ACC |
| ------------ | ---------------- | ------------------ | --- |
| SE7-32       | slowfast_opencv.py | slowfast_bm1684x_fp32_1b.bmodel |    0.633 |
| SE7-32       | slowfast_opencv.py | slowfast_bm1684x_fp32_4b.bmodel |    0.633 |
| SE7-32       | slowfast_opencv.py | slowfast_bm1684x_fp16_1b.bmodel |    0.633 |
| SE7-32       | slowfast_opencv.py | slowfast_bm1684x_fp16_4b.bmodel |    0.633 |
| SE7-32       | slowfast_opencv.py | slowfast_bm1684x_int8_1b.bmodel |    0.628 |
| SE7-32       | slowfast_opencv.py | slowfast_bm1684x_int8_4b.bmodel |    0.628 |
| SE7-32       | slowfast_opencv.soc | slowfast_bm1684x_fp32_1b.bmodel |    0.627 |
| SE7-32       | slowfast_opencv.soc | slowfast_bm1684x_fp32_4b.bmodel |    0.627 |
| SE7-32       | slowfast_opencv.soc | slowfast_bm1684x_fp16_1b.bmodel |    0.628 |
| SE7-32       | slowfast_opencv.soc | slowfast_bm1684x_fp16_4b.bmodel |    0.628 |
| SE7-32       | slowfast_opencv.soc | slowfast_bm1684x_int8_1b.bmodel |    0.628 |
| SE7-32       | slowfast_opencv.soc | slowfast_bm1684x_int8_4b.bmodel |    0.628 |
| SE9-16       | slowfast_opencv.py | slowfast_bm1688_fp32_1b.bmodel |    0.633 |
| SE9-16       | slowfast_opencv.py | slowfast_bm1688_fp32_4b.bmodel |    0.633 |
| SE9-16       | slowfast_opencv.py | slowfast_bm1688_fp16_1b.bmodel |    0.632 |
| SE9-16       | slowfast_opencv.py | slowfast_bm1688_fp16_4b.bmodel |    0.632 |
| SE9-16       | slowfast_opencv.py | slowfast_bm1688_int8_1b.bmodel |    0.628 |
| SE9-16       | slowfast_opencv.py | slowfast_bm1688_int8_4b.bmodel |    0.628 |
| SE9-16       | slowfast_opencv.soc | slowfast_bm1688_fp32_1b.bmodel |    0.627 |
| SE9-16       | slowfast_opencv.soc | slowfast_bm1688_fp32_4b.bmodel |    0.627 |
| SE9-16       | slowfast_opencv.soc | slowfast_bm1688_fp16_1b.bmodel |    0.627 |
| SE9-16       | slowfast_opencv.soc | slowfast_bm1688_fp16_4b.bmodel |    0.627 |
| SE9-16       | slowfast_opencv.soc | slowfast_bm1688_int8_1b.bmodel |    0.628 |
| SE9-16       | slowfast_opencv.soc | slowfast_bm1688_int8_4b.bmodel |    0.628 |
| SE9-16       | slowfast_opencv.py | slowfast_bm1688_fp32_1b_2core.bmodel |    0.633 |
| SE9-16       | slowfast_opencv.py | slowfast_bm1688_fp32_4b_2core.bmodel |    0.633 |
| SE9-16       | slowfast_opencv.py | slowfast_bm1688_fp16_1b_2core.bmodel |    0.632 |
| SE9-16       | slowfast_opencv.py | slowfast_bm1688_fp16_4b_2core.bmodel |    0.632 |
| SE9-16       | slowfast_opencv.py | slowfast_bm1688_int8_1b_2core.bmodel |    0.628 |
| SE9-16       | slowfast_opencv.py | slowfast_bm1688_int8_4b_2core.bmodel |    0.628 |
| SE9-16       | slowfast_opencv.soc | slowfast_bm1688_fp32_1b_2core.bmodel |    0.627 |
| SE9-16       | slowfast_opencv.soc | slowfast_bm1688_fp32_4b_2core.bmodel |    0.627 |
| SE9-16       | slowfast_opencv.soc | slowfast_bm1688_fp16_1b_2core.bmodel |    0.627 |
| SE9-16       | slowfast_opencv.soc | slowfast_bm1688_fp16_4b_2core.bmodel |    0.627 |
| SE9-16       | slowfast_opencv.soc | slowfast_bm1688_int8_1b_2core.bmodel |    0.628 |
| SE9-16       | slowfast_opencv.soc | slowfast_bm1688_int8_4b_2core.bmodel |    0.628 |
| SRM1-20      | slowfast_opencv.pcie | slowfast_bm1684x_fp32_1b.bmodel |    0.630 |
| SRM1-20      | slowfast_opencv.pcie | slowfast_bm1684x_fp32_4b.bmodel |    0.630 |
| SRM1-20      | slowfast_opencv.pcie | slowfast_bm1684x_fp16_1b.bmodel |    0.630 |
| SRM1-20      | slowfast_opencv.pcie | slowfast_bm1684x_fp16_4b.bmodel |    0.630 |
| SRM1-20      | slowfast_opencv.pcie | slowfast_bm1684x_int8_1b.bmodel |    0.621 |
| SRM1-20      | slowfast_opencv.pcie | slowfast_bm1684x_int8_4b.bmodel |    0.621 |

> **测试说明**：  
> 1. 由于sdk版本之间可能存在差异，实际运行结果与本表有<0.02的精度误差是正常的；
> 2. 在搭载了相同TPU和SOPHONSDK的PCIe或SoC平台上，相同程序的精度一致，SE7系列对应BM1684X，SE9系列中SE9-16对应BM1688；

## 7. 性能测试
### 7.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684X/slowfast_bm1684x_fp32_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是理论推理时间。
测试各个模型的理论推理时间，结果如下：

| 测试模型                       | calculate time(ms) |
| -------------------                |  -------------- |
| BM1684X/slowafst_bm1684x_fp32_1b.bmodel         |         199.57  |
| BM1684X/slowafst_bm1684x_fp32_4b.bmodel         |         193.86  |
| BM1684X/slowafst_bm1684x_fp16_1b.bmodel         |          33.07  |
| BM1684X/slowafst_bm1684x_fp16_4b.bmodel         |          31.86  |
| BM1684X/slowafst_bm1684x_int8_1b.bmodel         |          24.28  |
| BM1684X/slowafst_bm1684x_int8_4b.bmodel         |          23.84  |
| BM1688/slowafst_bm1688_fp32_1b.bmodel          |        1155.96  |
| BM1688/slowafst_bm1688_fp32_4b.bmodel          |        1142.72  |
| BM1688/slowafst_bm1688_fp16_1b.bmodel          |         223.08  |
| BM1688/slowafst_bm1688_fp16_4b.bmodel          |         217.67  |
| BM1688/slowafst_bm1688_int8_1b.bmodel          |          70.06  |
| BM1688/slowafst_bm1688_int8_4b.bmodel          |          66.22  |
| BM1688/slowafst_bm1688_fp32_1b_2core.bmodel    |         999.86  |
| BM1688/slowafst_bm1688_fp32_4b_2core.bmodel    |         984.33  |
| BM1688/slowafst_bm1688_fp16_1b_2core.bmodel    |         198.07  |
| BM1688/slowafst_bm1688_fp16_4b_2core.bmodel    |         193.83  |
| BM1688/slowafst_bm1688_int8_1b_2core.bmodel    |          53.89  |
| BM1688/slowafst_bm1688_int8_4b_2core.bmodel    |          50.96  |

> **测试说明**：  
1. 性能测试结果具有一定的波动性；
2. `calculate time`已折算为每个视频平均推理时间；
3. SoC和PCIe的测试结果基本一致。

### 7.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的视频解码时间、预处理时间、推理时间、后处理时间。C++和Python例程打印的时间已经折算为单张图片的处理时间。

在不同的测试平台上，使用不同的例程、模型测试`datasets/sampled_k400`，性能测试结果如下：
|    测试平台  |     测试程序  |      测试模型     |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ------------- | ---------------- | -------- | ---------   | ---------------| --------- |
|   SE7-32    |slowfast_opencv.py |slowfast_bm1684x_fp32_1b.bmodel|     128.55      |     535.26      |     280.79      |      0.28       |
|   SE7-32    |slowfast_opencv.py |slowfast_bm1684x_fp32_4b.bmodel|     128.06      |     597.35      |     284.46      |      0.14       |
|   SE7-32    |slowfast_opencv.py |slowfast_bm1684x_fp16_1b.bmodel|     129.23      |     534.43      |     114.50      |      0.28       |
|   SE7-32    |slowfast_opencv.py |slowfast_bm1684x_fp16_4b.bmodel|     127.40      |     595.62      |     121.54      |      0.14       |
|   SE7-32    |slowfast_opencv.py |slowfast_bm1684x_int8_1b.bmodel|     129.18      |     535.00      |     105.71      |      0.28       |
|   SE7-32    |slowfast_opencv.py |slowfast_bm1684x_int8_4b.bmodel|     128.09      |     596.31      |     114.01      |      0.13       |
|   SE7-32    |slowfast_opencv.soc|slowfast_bm1684x_fp32_1b.bmodel|      95.94      |     138.04      |     199.35      |      0.37       |
|   SE7-32    |slowfast_opencv.soc|slowfast_bm1684x_fp32_4b.bmodel|      96.54      |     137.84      |     193.76      |      0.35       |
|   SE7-32    |slowfast_opencv.soc|slowfast_bm1684x_fp16_1b.bmodel|      95.82      |     137.14      |      32.91      |      0.37       |
|   SE7-32    |slowfast_opencv.soc|slowfast_bm1684x_fp16_4b.bmodel|      95.93      |     136.93      |      31.75      |      0.35       |
|   SE7-32    |slowfast_opencv.soc|slowfast_bm1684x_int8_1b.bmodel|      95.68      |     138.00      |      24.19      |      0.37       |
|   SE7-32    |slowfast_opencv.soc|slowfast_bm1684x_int8_4b.bmodel|      95.88      |     137.42      |      23.81      |      0.34       |
|   SE9-16    |slowfast_opencv.py |slowfast_bm1688_fp32_1b.bmodel|     177.17      |     731.05      |     1256.12     |      0.41       |
|   SE9-16    |slowfast_opencv.py |slowfast_bm1688_fp32_4b.bmodel|     176.96      |     804.29      |     1254.24     |      0.20       |
|   SE9-16    |slowfast_opencv.py |slowfast_bm1688_fp16_1b.bmodel|     177.98      |     732.07      |     324.62      |      0.40       |
|   SE9-16    |slowfast_opencv.py |slowfast_bm1688_fp16_4b.bmodel|     177.27      |     803.49      |     329.01      |      0.20       |
|   SE9-16    |slowfast_opencv.py |slowfast_bm1688_int8_1b.bmodel|     176.73      |     730.13      |     171.32      |      0.39       |
|   SE9-16    |slowfast_opencv.py |slowfast_bm1688_int8_4b.bmodel|     177.92      |     805.44      |     177.44      |      0.19       |
|   SE9-16    |slowfast_opencv.soc|slowfast_bm1688_fp32_1b.bmodel|     117.47      |     174.13      |     1155.48     |      0.62       |
|   SE9-16    |slowfast_opencv.soc|slowfast_bm1688_fp32_4b.bmodel|     118.97      |     174.07      |     1142.85     |      0.65       |
|   SE9-16    |slowfast_opencv.soc|slowfast_bm1688_fp16_1b.bmodel|     116.81      |     174.06      |     222.16      |      0.61       |
|   SE9-16    |slowfast_opencv.soc|slowfast_bm1688_fp16_4b.bmodel|     119.03      |     174.26      |     217.44      |      0.53       |
|   SE9-16    |slowfast_opencv.soc|slowfast_bm1688_int8_1b.bmodel|     117.55      |     174.38      |      69.07      |      0.59       |
|   SE9-16    |slowfast_opencv.soc|slowfast_bm1688_int8_4b.bmodel|     118.12      |     174.30      |      66.01      |      0.55       |
|   SE9-16    |slowfast_opencv.py |slowfast_bm1688_fp32_1b_2core.bmodel|     177.62      |     731.70      |     1103.99     |      0.40       |
|   SE9-16    |slowfast_opencv.py |slowfast_bm1688_fp32_4b_2core.bmodel|     177.36      |     806.39      |     1095.97     |      0.20       |
|   SE9-16    |slowfast_opencv.py |slowfast_bm1688_fp16_1b_2core.bmodel|     177.74      |     730.70      |     299.17      |      0.41       |
|   SE9-16    |slowfast_opencv.py |slowfast_bm1688_fp16_4b_2core.bmodel|     176.96      |     806.46      |     305.09      |      0.20       |
|   SE9-16    |slowfast_opencv.py |slowfast_bm1688_int8_1b_2core.bmodel|     176.89      |     731.70      |     155.36      |      0.40       |
|   SE9-16    |slowfast_opencv.py |slowfast_bm1688_int8_4b_2core.bmodel|     177.62      |     804.52      |     162.30      |      0.19       |
|   SE9-16    |slowfast_opencv.soc|slowfast_bm1688_fp32_1b_2core.bmodel|     118.24      |     173.65      |     998.84      |      0.60       |
|   SE9-16    |slowfast_opencv.soc|slowfast_bm1688_fp32_4b_2core.bmodel|     119.54      |     173.59      |     984.25      |      0.91       |
|   SE9-16    |slowfast_opencv.soc|slowfast_bm1688_fp16_1b_2core.bmodel|     117.01      |     172.96      |     197.01      |      0.60       |
|   SE9-16    |slowfast_opencv.soc|slowfast_bm1688_fp16_4b_2core.bmodel|     119.55      |     173.87      |     193.42      |      0.56       |
|   SE9-16    |slowfast_opencv.soc|slowfast_bm1688_int8_1b_2core.bmodel|     117.10      |     173.77      |      52.90      |      0.59       |
|   SE9-16    |slowfast_opencv.soc|slowfast_bm1688_int8_4b_2core.bmodel|     118.87      |     173.69      |      50.75      |      0.56       |
|   SRM1-20   |slowfast_opencv.pcie|slowfast_bm1684x_fp32_1b.bmodel|      1316.19      |     626.92       |     232.19      |      0.58       |
|   SRM1-20   |slowfast_opencv.pcie|slowfast_bm1684x_fp32_4b.bmodel|      1399.07      |     685.75       |     224.80      |      0.51       |
|   SRM1-20   |slowfast_opencv.pcie|slowfast_bm1684x_fp16_1b.bmodel|      1324.40      |     619.56       |      38.29      |      0.56       |
|   SRM1-20   |slowfast_opencv.pcie|slowfast_bm1684x_fp16_4b.bmodel|      1372.36      |     648.29       |      36.15      |      0.50       |
|   SRM1-20   |slowfast_opencv.pcie|slowfast_bm1684x_int8_1b.bmodel|      1321.58      |     630.19       |      28.99      |      0.56       |
|   SRM1-20   |slowfast_opencv.pcie|slowfast_bm1684x_int8_4b.bmodel|      1372.79      |     679.48       |      27.56      |      0.49       |


> **测试说明**：  
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. SE7-32的主控处理器均为8核CA53@2.3GHz，SE9-16的主控处理器为8核CA53@1.6GHz，SE9-8为6核CA53@1.6GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异，不同的阈值对后处理时间影响较大。
> 5. SlowFast的后处理只有softmax，耗时很短，可以忽略。
> 6. riscv平台上，python例程的opencv目前不能处理视频。

## 8. FAQ
请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。
