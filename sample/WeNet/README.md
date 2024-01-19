# WeNet

## 目录

* [1. 简介](#1-简介)
* [2. 特性](#2-特性)
* [3. 准备模型与数据](#3-准备模型与数据)
* [4. 模型编译](#4-模型编译)
  * [4.1 TPU-MLIR编译BModel](#41-tpu-mlir编译bmodel)
* [5. 例程测试](#5-例程测试)
* [6. 精度测试](#6-精度测试)
  * [6.1 测试方法](#61-测试方法)
  * [6.2 测试结果](#62-测试结果)
* [7. 性能测试](#7-性能测试)
  * [7.1 bmrt_test](#71-bmrt_test)
  * [7.2 程序运行性能](#72-程序运行性能)
* [8. FAQ](#8-faq)

## 1. 简介
WeNet是一款面向工业落地应用的语音识别工具包，提供了从语音识别模型的训练到部署的一条龙服务。本例程对[WeNet官方开源仓库](https://github.com/wenet-e2e/wenet)中基于aishell的预训练模型和算法进行移植，使之能在SOPHON BM1684/BM1684X/BM1688上进行推理测试。后处理用到的ctc decoder代码来自[Ctc Decoder](https://github.com/Kevindurant111/ctcdecode-cpp.git)。

## 2. 特性
* 支持BM1688(SoC)、BM1684X(x86 PCIe、SoC)、BM1684(x86 PCIe、SoC)
* 支持FP32、FP16(BM1688/BM1684X)模型编译和推理
* 支持基于torchaudio的Python推理和基于Armadillo的C++推理
* 支持单batch模型推理
* 支持流式语音的测试

## 3. 准备模型与数据
建议使用TPU-MLIR编译BModel，Pytorch模型在编译前要导出成onnx模型。

​同时，您需要准备用于测试的数据集。

​本例程在`scripts`目录下提供了相关模型和数据的下载脚本`download.sh`，您也可以自己准备模型和数据集，并参考[4. 模型编译](#4-模型编译)进行模型转换。

```bash
# 安装unzip，若已安装请跳过
sudo apt install unzip
chmod -R +x scripts/

# 如果您只测试python例程，建议直接在目标平台（x86 PCIe、arm SoC）运行该下载脚本。
# 如果您需要测试cpp例程，对于SoC平台，cpp例程提供交叉编译的方式，需要在x86服务器上面运行该下载脚本。
./scripts/download.sh
```

下载的模型包括：
```
./models
├── BM1684
│   ├── wenet_decoder_fp32.bmodel             # 使用TPU-MLIR编译，用于BM1684的FP32 Decoder BModel，batch_size=1
│   └── wenet_encoder_fp32.bmodel             # 使用TPU-MLIR编译，用于BM1684的FP32 Encoder BModel，batch_size=1
├── BM1684X
│   ├── wenet_decoder_fp16.bmodel             # 使用TPU-MLIR编译，用于BM1684X的FP16 Decoder BModel，batch_size=1
│   ├── wenet_decoder_fp32.bmodel             # 使用TPU-MLIR编译，用于BM1684X的FP32 Decoder BModel，batch_size=1
│   ├── wenet_encoder_fp16.bmodel             # 使用TPU-MLIR编译，用于BM1684X的FP16 Encoder BModel，batch_size=1
│   └── wenet_encoder_fp32.bmodel             # 使用TPU-MLIR编译，用于BM1684X的FP32 Encoder BModel，batch_size=1
├── BM1688
│   ├── wenet_decoder_fp16.bmodel             # 使用TPU-MLIR编译，用于BM1688的FP16 Decoder BModel，batch_size=1，num_core=1
│   ├── wenet_decoder_fp32.bmodel             # 使用TPU-MLIR编译，用于BM1688的FP32 Decoder BModel，batch_size=1，num_core=1
│   ├── wenet_encoder_fp16.bmodel             # 使用TPU-MLIR编译，用于BM1688的FP16 Encoder BModel，batch_size=1，num_core=1
│   └── wenet_encoder_fp32.bmodel             # 使用TPU-MLIR编译，用于BM1688的FP32 Encoder BModel，batch_size=1，num_core=1
└── onnx
    ├── wenet_encoder.onnx                    # 导出的encoder onnx模型
    └── wenet_decoder.onnx                    # 导出的decoder onnx模型       
```

下载的数据包括：
```
./datasets
├── aishell_S0764                                      
    ├── *.wav                                 # 从aishell数据集中抽取的用于测试的音频文件
    ├── aishell_S0764.list                    # 数据集的描述文件
    └── ground_truth.txt                      # 数据集标签文件，用于计算精度评价指标  
```

下载的Python拓展模块包括：
```
./python/swig_decoders_x86_64                 # x86_64架构下编译好的swig_decoder模块
├── EGG-INFO                                       
├── _swig_decoders.py                              
├── swig_decoders.py                               
└── _swig_decoders.cpython-38-x86_64-linux-gnu.so               
./python/swig_decoders_aarch64                # aarch64架构下编译好的swig_decoder模块
├── EGG-INFO                                       
├── _swig_decoders.py                                
├── swig_decoders.py                               
└── _swig_decoders.cpython-38-aarch64-linux-gnu.so       

下载的Cpp交叉编译依赖包括：
./cpp/cross_compile_module
├──3rd_party                                  # aarch64架构下编译好的第三方库
└──ctcdecode-cpp                              # aarch64架构下编译好的ctcdecode-cpp
```

## 4. 模型编译
导出的模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。

### 4.1 TPU-MLIR编译BModel
导出的模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。建议使用TPU-MLIR编译BModel。

模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index/material/31/all.html)相应版本的SDK中获取)。

- 生成FP32 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684/BM1684X/BM1688**），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684 #bm1684x/bm1688
```

​执行上述命令会在`models/BM1684`等文件夹下生成`wenet_encoder_fp32.bmodel`和`wenet_decoder_fp32.bmodel`文件，即转换好的FP32 BModel。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688**），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x #bm1688
```

​执行上述命令会在`models/BM1684X/`等文件夹下生成`wenet_encoder_fp16.bmodel`和`wenet_decoder_fp16.bmodel`等文件，即转换好的FP16 BModel。


## 5. 例程测试
- [C++例程](./cpp/README.md)
- [Python例程](./python/README.md)

## 6. 精度测试
### 6.1 测试方法
首先，参考[Python例程](python/README.md#22-测试图片)推理要测试的数据集，生成预测的txt文件，注意修改数据集(datasets/aishell_S0764)和相关参数。  
然后，使用`tools`目录下的`eval_aishell.py`脚本，将测试生成的txt文件与测试集标签txt文件进行对比，计算出语音识别的评价指标，命令如下：
```bash
# 请根据实际情况修改程序路径和txt文件路径
python3 tools/eval_aishell.py --char=1 --v=1 datasets/aishell_S0764/ground_truth.txt python/result.txt  > online_wer
cat online_wer | grep "Overall"
```

### 6.2 测试结果
在aishell数据集上，精度测试结果如下：
|   测试平台    |    测试程序   |              测试模型                                 | WER    |
| ------------ | ------------ | ----------------------------------------------------- | ------ |
| BM1684 SoC   | wenet.py     | wenet_encoder_fp32.bmodel                             | 2.70%  |
| BM1684 SoC   | wenet.soc    | wenet_encoder_fp32.bmodel                             | 5.84%  |
| BM1684 SoC   | wenet.py     | wenet_encoder_fp32.bmodel + wenet_decoder_fp32.bmodel | 1.80%  |
| BM1684 SoC   | wenet.soc    | wenet_encoder_fp32.bmodel + wenet_decoder_fp32.bmodel | 3.45%  |
| BM1684X SoC  | wenet.py     | wenet_encoder_fp32.bmodel                             | 2.70%  | 
| BM1684X SoC  | wenet.soc    | wenet_encoder_fp32.bmodel                             | 2.55%  | 
| BM1684X SoC  | wenet.py     | wenet_encoder_fp32.bmodel + wenet_decoder_fp32.bmodel | 1.80%  | 
| BM1684X SoC  | wenet.soc    | wenet_encoder_fp32.bmodel + wenet_decoder_fp32.bmodel | 1.72%  | 
| BM1684X SoC  | wenet.py     | wenet_encoder_fp16.bmodel                             | 2.70%  | 
| BM1684X SoC  | wenet.soc    | wenet_encoder_fp16.bmodel                             | 2.55%  | 
| BM1684X SoC  | wenet.py     | wenet_encoder_fp16.bmodel + wenet_decoder_fp16.bmodel | 1.80%  | 
| BM1684X SoC  | wenet.soc    | wenet_encoder_fp16.bmodel + wenet_decoder_fp16.bmodel | 1.72%  | 
| BM1688 SoC   | wenet.py     | wenet_encoder_fp32.bmodel                             | 2.77%  | 
| BM1688 SoC   | wenet.py     | wenet_encoder_fp16.bmodel                             | 2.77%  | 
| BM1688 SoC   | wenet.py     | wenet_encoder_fp32.bmodel + wenet_decoder_fp32.bmodel | 1.87%  | 
| BM1688 SoC   | wenet.py     | wenet_encoder_fp16.bmodel + wenet_decoder_fp16.bmodel | 1.87%  | 
| BM1688 SoC   | wenet.soc    | wenet_encoder_fp32.bmodel                             | 2.77%  | 
| BM1688 SoC   | wenet.soc    | wenet_encoder_fp16.bmodel                             | 2.85%  | 
| BM1688 SoC   | wenet.soc    | wenet_encoder_fp32.bmodel + wenet_decoder_fp32.bmodel | 2.10%  | 
| BM1688 SoC   | wenet.soc    | wenet_encoder_fp16.bmodel + wenet_decoder_fp16.bmodel | 1.80%  | 

> **测试说明**：  
1. 在使用的模型相同的情况下，wer在不同的测试平台上是相同的。
2. 由于SDK版本之间的差异，实测的wer与本表有1%以内的差值是正常的。

## 7. 性能测试
### 7.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径
bmrt_test --bmodel models/BM1684/wenet_encoder_fp32.bmodel
```
测试结果中的`calculate time`就是模型推理的时间。
测试各个模型的理论推理时间，结果如下：
|                  测试模型                   | calculate time(ms) |
| ------------------------------------------- | -----------------|
| BM1684/wenet_encoder_fp32.bmodel            |  20.1            |
| BM1684/wenet_decoder_fp32.bmodel            |  761.3           |
| BM1684X/wenet_encoder_fp32.bmodel           |  9.5             |
| BM1684X/wenet_encoder_fp16.bmodel           |  3.2             |
| BM1684X/wenet_decoder_fp32.bmodel           |  307.7           |
| BM1684X/wenet_decoder_fp16.bmodel           |  74.2            |
| BM1688/wenet_encoder_fp32.bmodel            |  20.2            |
| BM1688/wenet_encoder_fp16.bmodel            |  7.2             |
| BM1688/wenet_decoder_fp32.bmodel            |  722.6           |
| BM1688/wenet_decoder_fp16.bmodel            |  230.6           |

> **测试说明**：  
1. 性能测试结果具有一定的波动性；
2. encoder的特征长度为67，对应为0.67s的音频；decoder的特征长度为350，对应为3.5s的音频。

### 7.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++例程打印的预处理时间、推理时间、后处理时间为整个batch处理的时间，需除以相应的batch size才是每张图片的处理时间。

在不同的测试平台上，使用数据集`datasets/aishell_S0764/aishell_S0764.list`，测试不同的例程和模型，性能测试结果如下：
|    测试平台  |  测试程序 |             测试模型                                   |preprocess_time|encoder_inference_time|decoder_inference_time|postprocess_time| 
| ----------- | --------- | ----------------------------------------------------- | ------------- | -------------------- | -------------------- | ----------------- |
| BM1684 SoC  | wenet.py  | wenet_encoder_fp32.bmodel                             | 15.65         | 24.40                |  none                | 39.76            |
| BM1684 SoC  | wenet.py  | wenet_encoder_fp32.bmodel + wenet_decoder_fp32.bmodel | 14.83         | 24.41                |  713.74              | 45.67            |
| BM1684 SoC  | wenet.soc | wenet_encoder_fp32.bmodel                             | 422.98        | 20.10                |  none                | 8.21             |
| BM1684 SoC  | wenet.soc | wenet_encoder_fp32.bmodel + wenet_decoder_fp32.bmodel | 423.75        | 20.11                |  713.78              | 13.99            |  
| BM1684x SoC | wenet.py  | wenet_encoder_fp32.bmodel                             | 22.28         | 14.24                |  none                | 40.18            |
| BM1684x SoC | wenet.py  | wenet_encoder_fp32.bmodel + wenet_decoder_fp32.bmodel | 19.29         | 14.24                |  309.04              | 47.86            |
| BM1684x SoC | wenet.soc | wenet_encoder_fp32.bmodel                             | 272.70        | 9.51                 |  none                | 8.29             |
| BM1684x SoC | wenet.soc | wenet_encoder_fp32.bmodel + wenet_decoder_fp32.bmodel | 272.10        | 9.50                 |  307.50              | 13.17            |
| BM1684x SoC | wenet.py  | wenet_encoder_fp16.bmodel                             | 15.18         | 7.98                 |  none                | 38.87            |
| BM1684x SoC | wenet.py  | wenet_encoder_fp16.bmodel + wenet_decoder_fp16.bmodel | 15.20         | 7.94                 |  74.74               | 47.42            |
| BM1684x SoC | wenet.soc | wenet_encoder_fp16.bmodel                             | 272.80        | 3.23                 |  none                | 10.42            |
| BM1684x SoC | wenet.soc | wenet_encoder_fp16.bmodel + wenet_decoder_fp16.bmodel | 272.47        | 3.24                 |  74.23               | 16.68            |  
| BM1688 SoC  | wenet.py  | wenet_encoder_fp32.bmodel                             | 25.29         | 25.99                |  none                | 54.84            |
| BM1688 SoC  | wenet.py  | wenet_encoder_fp32.bmodel + wenet_decoder_fp32.bmodel | 24.84         | 26.00                |  723.50              | 65.64            |
| BM1688 SoC  | wenet.soc | wenet_encoder_fp32.bmodel                             | 448.44        | 19.75                |  none                | 9.15             |
| BM1688 SoC  | wenet.soc | wenet_encoder_fp32.bmodel + wenet_decoder_fp32.bmodel | 448.66        | 19.73                |  721.53              | 16.86            |
| BM1688 SoC  | wenet.py  | wenet_encoder_fp16.bmodel                             | 27.59         | 13.08                |  none                | 57.07            |
| BM1688 SoC  | wenet.py  | wenet_encoder_fp16.bmodel + wenet_decoder_fp16.bmodel | 29.56         | 12.93                |  230.50              | 65.34            |
| BM1688 SoC  | wenet.soc | wenet_encoder_fp16.bmodel                             | 448.46        | 6.73                 |  none                | 8.70             |
| BM1688 SoC  | wenet.soc | wenet_encoder_fp16.bmodel + wenet_decoder_fp16.bmodel | 448.20        | 6.74                 |  228.98              | 16.15            |  


> **测试说明**：  
1. 时间单位均为毫秒(ms)，统计的时间均为1秒音频处理的时间(本例程用到的测试音频总时长442.955s)；
2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
3. BM1684/1684X SoC的主控处理器均为8核 ARM A53 42320 DMIPS @2.3GHz，PCIe上的性能由于处理器的不同可能存在较大差异；

## 8. FAQ  
1. ImportError: xxxx/libstdc++.so.6: version `GLIBCXX_3.4.30' not found: 常出现在pcie模式下，原因是编译好的ctc decoder与本机的环境不适配。  
解决方法：在要运行的主机上重新编译一份ctc decoder。
```bash
git clone https://github.com/Slyne/ctc_decoder.git  
sudo apt-get update
sudo apt-get install swig
sudo apt-get install python3-dev 
cd ctc_decoder/swig
sudo ./setup.sh
```  
2. bm_fft暂不支持1684x/1688，仅能在1684设备上使用。  
3. encoder与decoder的shape暂时无法调整，仅能编译和使用固定shape的bmodel，因此目前C++和Python例程的某些参数为固定参数。  

其他常见问题请参考[SOPHON-DEMO FAQ](../../docs/FAQ.md)。
