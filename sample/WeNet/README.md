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
* [9. WeNetSpeech适配](#9-WeNetSpeech适配)

## 1. 简介
WeNet是一款面向工业落地应用的语音识别工具包，提供了从语音识别模型的训练到部署的一条龙服务。本例程对[WeNet官方开源仓库](https://github.com/wenet-e2e/wenet)中基于aishell的预训练模型和算法进行移植，使之能在SOPHON BM1684/BM1684X/BM1688/CV186X上进行推理测试。后处理用到的ctc decoder代码来自[Ctc Decoder](https://github.com/Kevindurant111/ctcdecode-cpp.git)。

## 2. 特性
* 支持BM1688/CV186X(SoC)、BM1684X(x86 PCIe、SoC)、BM1684(x86 PCIe、SoC)
* 支持FP32、FP16(BM1688/BM1684X/CV186X)模型编译和推理
* 支持基于torchaudio的Python推理和基于Armadillo的C++推理
* 支持单batch模型推理
* 支持流式和非流式语音的测试

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
│   ├── wenet_decoder_fp32.bmodel               # 使用TPU-MLIR编译，用于BM1684的FP32 Decoder BModel，batch_size=1
│   ├── wenet_encoder_non_streaming_fp32.bmodel # 使用TPU-MLIR编译，用于BM1684的非流式FP32 Encoder BModel，batch_size=1
│   └── wenet_encoder_streaming_fp32.bmodel     # 使用TPU-MLIR编译，用于BM1684的流式FP32 Encoder BModel，batch_size=1
├── BM1684X
│   ├── wenet_decoder_fp16.bmodel               # 使用TPU-MLIR编译，用于BM1684X的FP16 Decoder BModel，batch_size=1     
│   ├── wenet_decoder_fp32.bmodel               # 使用TPU-MLIR编译，用于BM1684X的FP32 Decoder BModel，batch_size=1     
│   ├── wenet_encoder_non_streaming_fp16.bmodel # 使用TPU-MLIR编译，用于BM1684X的非流式FP16 Encoder BModel，batch_size=1
│   ├── wenet_encoder_non_streaming_fp32.bmodel # 使用TPU-MLIR编译，用于BM1684X的非流式FP32 Encoder BModel，batch_size=1
│   ├── wenet_encoder_streaming_fp16.bmodel     # 使用TPU-MLIR编译，用于BM1684X的流式FP16 Encoder BModel，batch_size=1  
│   └── wenet_encoder_streaming_fp32.bmodel     # 使用TPU-MLIR编译，用于BM1684X的流式FP32 Encoder BModel，batch_size=1  
├── BM1688
│   ├── wenet_decoder_fp16.bmodel               # 使用TPU-MLIR编译，用于BM1688的FP16 Decoder BModel，batch_size=1     
│   ├── wenet_decoder_fp32.bmodel               # 使用TPU-MLIR编译，用于BM1688的FP32 Decoder BModel，batch_size=1   
│   ├── wenet_encoder_non_streaming_fp16.bmodel # 使用TPU-MLIR编译，用于BM1688的非流式FP16 Encoder BModel，batch_size=1
│   ├── wenet_encoder_non_streaming_fp32.bmodel # 使用TPU-MLIR编译，用于BM1688的非流式FP32 Encoder BModel，batch_size=1
│   ├── wenet_encoder_streaming_fp16.bmodel     # 使用TPU-MLIR编译，用于BM1688的流式FP16 Encoder BModel，batch_size=1 
│   └── wenet_encoder_streaming_fp32.bmodel     # 使用TPU-MLIR编译，用于BM1688的流式FP32 Encoder BModel，batch_size=1 
├── CV186X
│   ├── wenet_decoder_fp16.bmodel               # 使用TPU-MLIR编译，用于CV186X的FP16 Decoder BModel，batch_size=1     
│   ├── wenet_decoder_fp32.bmodel               # 使用TPU-MLIR编译，用于CV186X的FP32 Decoder BModel，batch_size=1         
│   ├── wenet_encoder_non_streaming_fp16.bmodel # 使用TPU-MLIR编译，用于CV186X的非流式FP16 Encoder BModel，batch_size=1    
│   ├── wenet_encoder_non_streaming_fp32.bmodel # 使用TPU-MLIR编译，用于CV186X的非流式FP32 Encoder BModel，batch_size=1  
│   ├── wenet_encoder_streaming_fp16.bmodel     # 使用TPU-MLIR编译，用于CV186X的流式FP16 Encoder BModel，batch_size=1 
│   └── wenet_encoder_streaming_fp32.bmodel     # 使用TPU-MLIR编译，用于CV186X的流式FP32 Encoder BModel，batch_size=1 
└── onnx
    ├── wenet_decoder.onnx                      # 导出的流式decoder onnx模型
    ├── wenet_decoder_qtable                    # 转fp16的decoder时，传给model_deploy的混精度敏感层
    ├── wenet_encoder_non_streaming.onnx        # 导出的非流式encoder onnx模型
    └── wenet_encoder_streaming.onnx            # 导出的流式encoder onnx模型
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

下载的Cpp编译依赖包括：
./cpp/ctcdecode-cpp                           # x86架构下编译好的ctcdecode-cpp(基于ubuntu20.04, gcc9.4.0)
./cpp/cross_compile_module
├──3rd_party                                  # aarch64架构下编译好的第三方库
└──ctcdecode-cpp                              # aarch64架构下编译好的ctcdecode-cpp
```

## 4. 模型编译
导出的模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。

### 4.1 TPU-MLIR编译BModel
导出的模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。建议使用TPU-MLIR编译BModel。

模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的SDK中获取)。

- 生成FP32 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684/BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684 #bm1684x/bm1688/cv186x
```

​执行上述命令会在`models/BM1684`等文件夹下生成`wenet_encoder_fp32.bmodel`和`wenet_decoder_fp32.bmodel`文件，即转换好的FP32 BModel。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x #bm1688/cv186x
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
| SE5-16       | wenet.py           | wenet_encoder_streaming_fp32.bmodel                                    |    2.700% |
| SE5-16       | wenet.py           | wenet_encoder_streaming_fp32.bmodel + wenet_decoder_fp32.bmodel        |    1.800% |
| SE5-16       | wenet.soc          | wenet_encoder_streaming_fp32.bmodel                                    |    2.550% |
| SE5-16       | wenet.soc          | wenet_encoder_streaming_fp32.bmodel + wenet_decoder_fp32.bmodel        |    1.720% |
| SE5-16       | wenet.py           | wenet_encoder_non_streaming_fp32.bmodel                                |    2.020% |
| SE5-16       | wenet.py           | wenet_encoder_non_streaming_fp32.bmodel + wenet_decoder_fp32.bmodel    |    1.650% |
| SE5-16       | wenet.soc          | wenet_encoder_non_streaming_fp32.bmodel                                |    2.020% |
| SE5-16       | wenet.soc          | wenet_encoder_non_streaming_fp32.bmodel + wenet_decoder_fp32.bmodel    |    1.650% |
| SE7-32       | wenet.py           | wenet_encoder_streaming_fp32.bmodel                                    |    2.700% |
| SE7-32       | wenet.py           | wenet_encoder_streaming_fp32.bmodel + wenet_decoder_fp32.bmodel        |    1.800% |
| SE7-32       | wenet.soc          | wenet_encoder_streaming_fp32.bmodel                                    |    2.550% |
| SE7-32       | wenet.soc          | wenet_encoder_streaming_fp32.bmodel + wenet_decoder_fp32.bmodel        |    1.720% |
| SE7-32       | wenet.py           | wenet_encoder_streaming_fp16.bmodel                                    |    2.700% |
| SE7-32       | wenet.py           | wenet_encoder_streaming_fp16.bmodel + wenet_decoder_fp16.bmodel        |    3.300% |
| SE7-32       | wenet.soc          | wenet_encoder_streaming_fp16.bmodel                                    |    2.550% |
| SE7-32       | wenet.soc          | wenet_encoder_streaming_fp16.bmodel + wenet_decoder_fp16.bmodel        |    3.150% |
| SE7-32       | wenet.py           | wenet_encoder_non_streaming_fp32.bmodel                                |    2.020% |
| SE7-32       | wenet.py           | wenet_encoder_non_streaming_fp32.bmodel + wenet_decoder_fp32.bmodel    |    1.650% |
| SE7-32       | wenet.soc          | wenet_encoder_non_streaming_fp32.bmodel                                |    2.020% |
| SE7-32       | wenet.soc          | wenet_encoder_non_streaming_fp32.bmodel + wenet_decoder_fp32.bmodel    |    1.650% |
| SE7-32       | wenet.py           | wenet_encoder_non_streaming_fp16.bmodel                                |    2.020% |
| SE7-32       | wenet.py           | wenet_encoder_non_streaming_fp16.bmodel + wenet_decoder_fp16.bmodel    |    2.620% |
| SE7-32       | wenet.soc          | wenet_encoder_non_streaming_fp16.bmodel                                |    2.020% |
| SE7-32       | wenet.soc          | wenet_encoder_non_streaming_fp16.bmodel + wenet_decoder_fp16.bmodel    |    2.620% |
| SE9-16       | wenet.py           | wenet_encoder_streaming_fp32.bmodel                                    |    2.770% |
| SE9-16       | wenet.py           | wenet_encoder_streaming_fp32.bmodel + wenet_decoder_fp32.bmodel        |    1.870% |
| SE9-16       | wenet.soc          | wenet_encoder_streaming_fp32.bmodel                                    |    2.700% |
| SE9-16       | wenet.soc          | wenet_encoder_streaming_fp32.bmodel + wenet_decoder_fp32.bmodel        |    2.100% |
| SE9-16       | wenet.py           | wenet_encoder_streaming_fp16.bmodel                                    |    2.700% |
| SE9-16       | wenet.py           | wenet_encoder_streaming_fp16.bmodel + wenet_decoder_fp16.bmodel        |    3.750% |
| SE9-16       | wenet.soc          | wenet_encoder_streaming_fp16.bmodel                                    |    2.550% |
| SE9-16       | wenet.soc          | wenet_encoder_streaming_fp16.bmodel + wenet_decoder_fp16.bmodel        |    3.820% |
| SE9-16       | wenet.py           | wenet_encoder_non_streaming_fp32.bmodel                                |    2.100% |
| SE9-16       | wenet.py           | wenet_encoder_non_streaming_fp32.bmodel + wenet_decoder_fp32.bmodel    |    2.100% |
| SE9-16       | wenet.soc          | wenet_encoder_non_streaming_fp32.bmodel                                |    2.100% |
| SE9-16       | wenet.soc          | wenet_encoder_non_streaming_fp32.bmodel + wenet_decoder_fp32.bmodel    |    2.100% |
| SE9-16       | wenet.py           | wenet_encoder_non_streaming_fp16.bmodel                                |    2.020% |
| SE9-16       | wenet.py           | wenet_encoder_non_streaming_fp16.bmodel + wenet_decoder_fp16.bmodel    |    2.770% |
| SE9-16       | wenet.soc          | wenet_encoder_non_streaming_fp16.bmodel                                |    2.020% |
| SE9-16       | wenet.soc          | wenet_encoder_non_streaming_fp16.bmodel + wenet_decoder_fp16.bmodel    |    2.770% |
| SE9-8        | wenet.py           | wenet_encoder_streaming_fp32.bmodel                                    |    2.770% |
| SE9-8        | wenet.py           | wenet_encoder_streaming_fp32.bmodel + wenet_decoder_fp32.bmodel        |    1.870% |
| SE9-8        | wenet.soc          | wenet_encoder_streaming_fp32.bmodel                                    |    2.700% |
| SE9-8        | wenet.soc          | wenet_encoder_streaming_fp32.bmodel + wenet_decoder_fp32.bmodel        |    2.100% |
| SE9-8        | wenet.py           | wenet_encoder_streaming_fp16.bmodel                                    |    2.700% |
| SE9-8        | wenet.py           | wenet_encoder_streaming_fp16.bmodel + wenet_decoder_fp16.bmodel        |    3.750% |
| SE9-8        | wenet.soc          | wenet_encoder_streaming_fp16.bmodel                                    |    2.550% |
| SE9-8        | wenet.soc          | wenet_encoder_streaming_fp16.bmodel + wenet_decoder_fp16.bmodel        |    3.820% |
| SE9-8        | wenet.py           | wenet_encoder_non_streaming_fp32.bmodel                                |    2.100% |
| SE9-8        | wenet.py           | wenet_encoder_non_streaming_fp32.bmodel + wenet_decoder_fp32.bmodel    |    2.100% |
| SE9-8        | wenet.soc          | wenet_encoder_non_streaming_fp32.bmodel                                |    2.100% |
| SE9-8        | wenet.soc          | wenet_encoder_non_streaming_fp32.bmodel + wenet_decoder_fp32.bmodel    |    2.100% |
| SE9-8        | wenet.py           | wenet_encoder_non_streaming_fp16.bmodel                                |    2.020% |
| SE9-8        | wenet.py           | wenet_encoder_non_streaming_fp16.bmodel + wenet_decoder_fp16.bmodel    |    2.770% |
| SE9-8        | wenet.soc          | wenet_encoder_non_streaming_fp16.bmodel                                |    2.020% |
| SE9-8        | wenet.soc          | wenet_encoder_non_streaming_fp16.bmodel + wenet_decoder_fp16.bmodel    |    2.770% |

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
| BM1684/wenet_encoder_streaming_fp32.bmodel        |          23.63  |
| BM1684/wenet_encoder_non_streaming_fp32.bmodel    |         138.78  |
| BM1684/wenet_decoder_fp32.bmodel                  |         941.14  |
| BM1684X/wenet_encoder_streaming_fp32.bmodel       |           9.79  |
| BM1684X/wenet_encoder_non_streaming_fp32.bmodel   |          71.95  |
| BM1684X/wenet_decoder_fp32.bmodel                 |         307.50  |
| BM1684X/wenet_encoder_streaming_fp16.bmodel       |           3.43  |
| BM1684X/wenet_encoder_non_streaming_fp16.bmodel   |          14.23  |
| BM1684X/wenet_decoder_fp16.bmodel                 |          62.54  |
| BM1688/wenet_encoder_streaming_fp32.bmodel        |          19.84  |
| BM1688/wenet_encoder_non_streaming_fp32.bmodel    |         211.54  |
| BM1688/wenet_decoder_fp32.bmodel                  |         722.56  |
| BM1688/wenet_encoder_streaming_fp16.bmodel        |           6.71  |
| BM1688/wenet_encoder_non_streaming_fp16.bmodel    |          44.37  |
| BM1688/wenet_decoder_fp16.bmodel                  |         179.78  |
| CV186X/wenet_encoder_streaming_fp32.bmodel        |          19.96  |
| CV186X/wenet_encoder_non_streaming_fp32.bmodel    |         212.00  |
| CV186X/wenet_decoder_fp32.bmodel                  |         722.76  |
| CV186X/wenet_encoder_streaming_fp16.bmodel        |           6.87  |
| CV186X/wenet_encoder_non_streaming_fp16.bmodel    |          44.21  |
| CV186X/wenet_decoder_fp16.bmodel                  |         177.34  |

> **测试说明**：  
> 1. 性能测试结果具有一定的波动性；
> 2. 流式encoder的特征长度为67，非流式encoder的特征长度为1200，decoder的特征长度为350。
> 3. 在搭载了相同TPU和SOPHONSDK的PCIe或SoC平台上，相同程序的精度一致，SE5系列对应BM1684，SE7系列对应BM1684X，SE9系列中，SE9-16对应BM1688，SE9-8对应CV186X；

### 7.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++例程打印的预处理时间、推理时间、后处理时间为整个batch处理的时间，需除以相应的batch size才是每张图片的处理时间。

在不同的测试平台上，使用数据集`datasets/aishell_S0764/aishell_S0764.list`，测试不同的例程和模型，性能测试结果如下：
|    测试平台  |  测试程序 |             测试模型                                   |preprocess_time|encoder_inference_time|decoder_inference_time|postprocess_time| 
| ----------- | --------- | ----------------------------------------------------- | ------------- | -------------------- | -------------------- | ----------------- |
|   SE5-16    |     wenet.py      |                 wenet_encoder_streaming_fp32.bmodel                  |      4.02       |      46.34      |      none       |      8.70       |
|   SE5-16    |     wenet.py      |   wenet_encoder_streaming_fp32.bmodel + wenet_decoder_fp32.bmodel    |      3.96       |      46.11      |     186.68      |      10.14      |
|   SE5-16    |     wenet.soc     |                 wenet_encoder_streaming_fp32.bmodel                  |      88.20      |      38.71      |      none       |      1.12       |
|   SE5-16    |     wenet.soc     |   wenet_encoder_streaming_fp32.bmodel + wenet_decoder_fp32.bmodel    |      88.95      |      38.72      |     186.51      |      1.29       |
|   SE5-16    |     wenet.py      |               wenet_encoder_non_streaming_fp32.bmodel                |      3.35       |      30.70      |      none       |      1.53       |
|   SE5-16    |     wenet.py      | wenet_encoder_non_streaming_fp32.bmodel + wenet_decoder_fp32.bmodel  |      3.11       |      30.77      |     186.99      |      3.34       |
|   SE5-16    |     wenet.soc     |               wenet_encoder_non_streaming_fp32.bmodel                |      88.09      |      29.60      |      none       |      0.36       |
|   SE5-16    |     wenet.soc     | wenet_encoder_non_streaming_fp32.bmodel + wenet_decoder_fp32.bmodel  |      88.92      |      29.62      |     186.58      |      0.46       |
|   SE7-32    |     wenet.py      |                 wenet_encoder_streaming_fp32.bmodel                  |      3.32       |      23.69      |      none       |      8.69       |
|   SE7-32    |     wenet.py      |   wenet_encoder_streaming_fp32.bmodel + wenet_decoder_fp32.bmodel    |      3.24       |      23.70      |      66.98      |      10.34      |
|   SE7-32    |     wenet.soc     |                 wenet_encoder_streaming_fp32.bmodel                  |      25.73      |      15.64      |      none       |      0.99       |
|   SE7-32    |     wenet.soc     |   wenet_encoder_streaming_fp32.bmodel + wenet_decoder_fp32.bmodel    |      26.43      |      15.69      |      66.64      |      1.09       |
|   SE7-32    |     wenet.py      |                 wenet_encoder_streaming_fp16.bmodel                  |      3.70       |      13.00      |      none       |      8.64       |
|   SE7-32    |     wenet.py      |   wenet_encoder_streaming_fp16.bmodel + wenet_decoder_fp16.bmodel    |      3.86       |      13.07      |      13.89      |      10.54      |
|   SE7-32    |     wenet.soc     |                 wenet_encoder_streaming_fp16.bmodel                  |      25.73      |      5.13       |      none       |      1.00       |
|   SE7-32    |     wenet.soc     |   wenet_encoder_streaming_fp16.bmodel + wenet_decoder_fp16.bmodel    |      26.43      |      5.10       |      13.55      |      1.07       |
|   SE7-32    |     wenet.py      |               wenet_encoder_non_streaming_fp32.bmodel                |      3.04       |      16.72      |      none       |      1.57       |
|   SE7-32    |     wenet.py      | wenet_encoder_non_streaming_fp32.bmodel + wenet_decoder_fp32.bmodel  |      3.14       |      16.71      |      66.97      |      3.39       |
|   SE7-32    |     wenet.soc     |               wenet_encoder_non_streaming_fp32.bmodel                |      25.77      |      15.53      |      none       |      0.32       |
|   SE7-32    |     wenet.soc     | wenet_encoder_non_streaming_fp32.bmodel + wenet_decoder_fp32.bmodel  |      26.24      |      15.53      |      66.63      |      0.40       |
|   SE7-32    |     wenet.py      |               wenet_encoder_non_streaming_fp16.bmodel                |      3.07       |      4.19       |      none       |      1.61       |
|   SE7-32    |     wenet.py      | wenet_encoder_non_streaming_fp16.bmodel + wenet_decoder_fp16.bmodel  |      3.29       |      4.19       |      13.87      |      3.30       |
|   SE7-32    |     wenet.soc     |               wenet_encoder_non_streaming_fp16.bmodel                |      25.76      |      3.03       |      none       |      0.51       |
|   SE7-32    |     wenet.soc     | wenet_encoder_non_streaming_fp16.bmodel + wenet_decoder_fp16.bmodel  |      26.50      |      3.02       |      13.55      |      0.59       |
|   SE9-16    |     wenet.py      |                 wenet_encoder_streaming_fp32.bmodel                  |      5.25       |      42.66      |      none       |      12.07      |
|   SE9-16    |     wenet.py      |   wenet_encoder_streaming_fp32.bmodel + wenet_decoder_fp32.bmodel    |      5.69       |      42.44      |     157.15      |      14.48      |
|   SE9-16    |     wenet.soc     |                 wenet_encoder_streaming_fp32.bmodel                  |      35.87      |      32.08      |      none       |      1.82       |
|   SE9-16    |     wenet.soc     |   wenet_encoder_streaming_fp32.bmodel + wenet_decoder_fp32.bmodel    |      36.90      |      32.09      |     156.63      |      1.99       |
|   SE9-16    |     wenet.py      |                 wenet_encoder_streaming_fp16.bmodel                  |      4.55       |      20.89      |      none       |      12.10      |
|   SE9-16    |     wenet.py      |   wenet_encoder_streaming_fp16.bmodel + wenet_decoder_fp16.bmodel    |      6.04       |      20.85      |      39.46      |      14.38      |
|   SE9-16    |     wenet.soc     |                 wenet_encoder_streaming_fp16.bmodel                  |      35.87      |      10.46      |      none       |      1.55       |
|   SE9-16    |     wenet.soc     |   wenet_encoder_streaming_fp16.bmodel + wenet_decoder_fp16.bmodel    |      36.83      |      10.48      |      39.01      |      1.69       |
|   SE9-16    |     wenet.py      |               wenet_encoder_non_streaming_fp32.bmodel                |      5.59       |      47.30      |      none       |      2.19       |
|   SE9-16    |     wenet.py      | wenet_encoder_non_streaming_fp32.bmodel + wenet_decoder_fp32.bmodel  |      5.32       |      47.24      |     157.15      |      4.68       |
|   SE9-16    |     wenet.soc     |               wenet_encoder_non_streaming_fp32.bmodel                |      35.97      |      45.74      |      none       |      0.62       |
|   SE9-16    |     wenet.soc     | wenet_encoder_non_streaming_fp32.bmodel + wenet_decoder_fp32.bmodel  |      36.77      |      45.74      |     156.65      |      0.75       |
|   SE9-16    |     wenet.py      |               wenet_encoder_non_streaming_fp16.bmodel                |      4.47       |      11.12      |      none       |      2.17       |
|   SE9-16    |     wenet.py      | wenet_encoder_non_streaming_fp16.bmodel + wenet_decoder_fp16.bmodel  |      4.38       |      11.03      |      39.48      |      4.63       |
|   SE9-16    |     wenet.soc     |               wenet_encoder_non_streaming_fp16.bmodel                |      35.88      |      9.53       |      none       |      0.49       |
|   SE9-16    |     wenet.soc     | wenet_encoder_non_streaming_fp16.bmodel + wenet_decoder_fp16.bmodel  |      36.65      |      9.53       |      39.01      |      0.61       |
|    SE9-8    |     wenet.py      |                 wenet_encoder_streaming_fp32.bmodel                  |      5.81       |      42.55      |      none       |      11.77      |
|    SE9-8    |     wenet.py      |   wenet_encoder_streaming_fp32.bmodel + wenet_decoder_fp32.bmodel    |      5.46       |      42.47      |     157.13      |      14.16      |
|    SE9-8    |     wenet.soc     |                 wenet_encoder_streaming_fp32.bmodel                  |      35.94      |      32.17      |      none       |      1.72       |
|    SE9-8    |     wenet.soc     |   wenet_encoder_streaming_fp32.bmodel + wenet_decoder_fp32.bmodel    |      36.96      |      32.21      |     156.64      |      1.88       |
|    SE9-8    |     wenet.py      |                 wenet_encoder_streaming_fp16.bmodel                  |      6.50       |      20.65      |      none       |      12.07      |
|    SE9-8    |     wenet.py      |   wenet_encoder_streaming_fp16.bmodel + wenet_decoder_fp16.bmodel    |      5.54       |      20.66      |      38.89      |      14.59      |
|    SE9-8    |     wenet.soc     |                 wenet_encoder_streaming_fp16.bmodel                  |      35.94      |      10.42      |      none       |      1.54       |
|    SE9-8    |     wenet.soc     |   wenet_encoder_streaming_fp16.bmodel + wenet_decoder_fp16.bmodel    |      36.77      |      10.44      |      38.42      |      1.65       |
|    SE9-8    |     wenet.py      |               wenet_encoder_non_streaming_fp32.bmodel                |      5.31       |      47.39      |      none       |      2.16       |
|    SE9-8    |     wenet.py      | wenet_encoder_non_streaming_fp32.bmodel + wenet_decoder_fp32.bmodel  |      5.19       |      47.38      |     157.14      |      4.59       |
|    SE9-8    |     wenet.soc     |               wenet_encoder_non_streaming_fp32.bmodel                |      35.90      |      45.87      |      none       |      0.62       |
|    SE9-8    |     wenet.soc     | wenet_encoder_non_streaming_fp32.bmodel + wenet_decoder_fp32.bmodel  |      36.77      |      45.87      |     156.64      |      0.76       |
|    SE9-8    |     wenet.py      |               wenet_encoder_non_streaming_fp16.bmodel                |      4.43       |      10.97      |      none       |      2.13       |
|    SE9-8    |     wenet.py      | wenet_encoder_non_streaming_fp16.bmodel + wenet_decoder_fp16.bmodel  |      4.40       |      10.97      |      38.89      |      4.68       |
|    SE9-8    |     wenet.soc     |               wenet_encoder_non_streaming_fp16.bmodel                |      35.91      |      9.48       |      none       |      0.49       |
|    SE9-8    |     wenet.soc     | wenet_encoder_non_streaming_fp16.bmodel + wenet_decoder_fp16.bmodel  |      36.67      |      9.49       |      38.42      |      0.63       |

> **测试说明**：  
> 1. 时间单位均为毫秒(ms)，统计的时间均为1秒音频处理的时间(本例程用到的测试音频总时长442.955s)；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. SE5-16/SE7-32的主控处理器均为8核CA53@2.3GHz，SE9-16为8核CA53@1.6GHz，SE9-8为6核CA53@1.6GHz，PCIe上的性能由于处理器的不同可能存在较大差异；

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

## 9. WeNetSpeech适配
除了基于Aishell的源模型，本例程也提供了基于WeNetSpeech源模型的适配方法，见[WeNetSpeech_Guide](./docs/WeNetSpeech_Guide.md)。