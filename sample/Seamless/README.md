# Seamless <!-- omit in toc -->

## 目录 <!-- omit in toc -->
- [1. 简介](#1-简介)
- [2. 特性](#2-特性)
- [3. 准备模型与数据](#3-准备模型与数据)
- [4. 模型编译](#4-模型编译)
  - [4.1 TPU-MLIR环境搭建](#41-tpu-mlir环境搭建)
    - [4.1.1 安装docker](#411-安装docker)
    - [4.1.2 下载并解压TPU-MLIR](#412-下载并解压tpu-mlir)
    - [4.1.3 创建并进入docker](#413-创建并进入docker)
  - [4.2 获取onnx](#42-获取onnx)
  - [4.3 bmodel编译](#43-bmodel编译)
- [5. 例程测试](#5-例程测试)
- [6. 精度测试](#6-精度测试)
  - [6.1 测试方法](#61-测试方法)
  - [6.2 测试结果](#62-测试结果)
- [7. 性能测试](#7-性能测试)

## 1. 简介
Seamless 是一个开源的深度学习语音识别模型，由 Meta 开发，它能够实现实时、多语言的语音识别、翻译，并支持跨多种环境和设备的灵活部署。本例程对[Seamless官方开源仓库](https://github.com/facebookresearch/seamless_communication)中的SeamlessStreaming算法进行移植，使之能在SOPHON BM1684X上进行推理。

## 2. 特性
* 支持BM1684X(x86 PCIe, SoC)
* 支持FP16(BM1684X)和FP32(BM1684X)模型编译和推理
* 支持基于SAIL推理的Python例程

## 3. 准备模型与数据
该模型目前只支持在1684X上运行，已提供编译好的bmodel，​同时，您需要准备用于测试的数据集。

​本例程在`scripts`目录下提供了相关模型和数据的下载脚本`download.sh`。

```bash
# 安装unzip，若已安装请跳过
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```

下载的模型包括：
```
./models
├── BM1684X
|   ├── seamless_streaming_encoder_frontend_fp16_s2t.bmodel                                          # SeamlessStreaming(s2t任务) Encoder的前端模块，fp16 BModel
|   ├── seamless_streaming_decoder_frontend_step_equal_1_fp16_s2t.bmodel                             # SeamlessStreaming(s2t任务) Decoder的前端模块，第一步解码的fp16 BModel
|   ├── seamless_streaming_decoder_final_proj_fp16_s2t.bmodel                                        # SeamlessStreaming(s2t任务) Decoder的线性模块，fp16 BModel
|   ├── seamless_streaming_decoder_frontend_step_bigger_1_fp16_s2t.bmodel                            # SeamlessStreaming(s2t任务) Decoder的前端模块，大于第一步解码的fp16 BModel
|   ├── seamless_streaming_encoder_fp16_s2t.bmodel                                                   # SeamlessStreaming(s2t任务) Encoder模型，fp16 BModel
|   ├── seamless_streaming_decoder_step_bigger_1_fp16_s2t.bmodel                                     # SeamlessStreaming(s2t任务) Decoder模块，大于第一步解码的fp16 BModel
|   └── seamless_streaming_decoder_step_equal_1_fp32_s2t.bmodel                                      # SeamlessStreaming(s2t任务) Decoder模块，第一步解码的fp32 BModel
└── tokenizer.model                                                                                  # SeamlessStreaming(s2t任务) tokenizer
```

下载的数据包括：
```
./datasets
|── aishell_S0764                             # 从aishell数据集中抽取的用于测试的音频文件
|   └── *.wav
├── aishell_S0764.list                        # 从aishell数据集的文件列表
├── ground_truth.txt                          # 从aishell数据集的预测真实值
└── test                                      # 测试使用的音频文件
    └── demo.wav
```
## 4. 模型编译
导出的模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。使用TPU-MLIR编译BModel。

模型编译前需要安装TPU-MLIR，需要参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)并替换`3. 安装TPU-MLIR`步骤中安装的TPU-MLIR为如下下载的。
```bash
python3 -m dfss --url=open@sophgo.com:sophon-demo/Seamless/tpu-mlir_seamless_streaming_s2t.tar.gz
tar -zxvf tpu-mlir_seamless_streaming_s2t.tar.gz
```

安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index/material/31/all.html)相应版本的SDK中获取)。

- 生成BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译BModel的脚本，请注意修改`gen_streaming_s2t_bmodel.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，如：

```bash
cd ./scripts
./gen_streaming_s2t_bmodel.sh
```

​执行上述命令会在`models/BM1684X`文件夹下生成`seamless_streaming_encoder_fp16_s2t.bmodel `等文件，即转换好的BModel。

## 5. 例程测试

- [Python例程](./python/README.md)

## 6. 精度测试
### 6.1 测试方法
首先，参考[Python例程](python/README.md#22-使用方式)进行中文语音到中文文字的转换，生成预测结果至result路径，注意修改数据集为../datasets/aishell_S0764和相关参数。

然后使用`tools`目录下的`eval_aishell.py`脚本，将测试生成的txt文件与测试集标签txt文件进行对比，计算出语音识别的评价指标，命令如下：
```bash
# 请根据实际情况修改程序路径和txt文件路径
python3 tools/eval_aishell.py --char=1 --v=1 datasets/ground_truth.txt python/results  > online_wer
cat online_wer | grep "Overall"
```
> **注意**：
> 1. 若遇到报错`OSError: libopencc.so.1: cannot open shared object file: No such file or directory`，需要安装依赖库`opencc`，执行`sudo apt-get install opencc`

### 6.2 测试结果
在aishell数据集上，ASR任务精度测试结果如下：
|   测试平台    |             测试程序                 |              测试模型                                  | WER    |
| ------------ | ------------------------------------ | ----------------------------------------------------- | ------ |
|   SE7-32     | pipeline_seamless_streaming_s2t.py   |     SeamlessStreaming(s2t任务)模型                     | 5.42%  |

> **测试说明**：
> 1. 在使用的模型相同的情况下，wer在不同的测试平台上是相同的。
> 2. 由于SDK版本之间的差异，实测的wer与本表有1%以内的差值是正常的。
> 3. 需要设置运行参数`--source_segment_size=640`。

## 7. 性能测试
|    测试平台   |              测试程序                 |           测试模型                  |  Decode time(ms) |  Preprocess time(ms)    |    Inference time(ms)   |
| -----------  | ------------------------------------- | -----------------------------------| ---------------- | ----------------------- | ----------------------- |
|   SE7-32     | pipeline_seamless_streaming_s2t.py    |   SeamlessStreaming(s2t任务)模型    | 39.86            |  7.40                   |  745.07                 |

> **测试说明**：
> 1. 该性能使用datasets/test/demo.wav音频进行测试，执行ASR任务，计算后得出平均每秒音频所需推理时间。
> 2. seamless模型的预处理主要包括加载语音，特征提取等，推理后的结果可直接转换为自然语言，时间可忽略不计，因此后处理部分时间统计到推理部分。
> 3. 性能测试结果具有一定的波动性，实测结果与该表结果有误差属正常现象，建议多次测试取平均值。
> 4. BM1684X SoC的主控处理器为8核 ARM A53 42320 DMIPS @2.3GHz。
> 5. 需要设置运行参数`--source_segment_size=640`。

