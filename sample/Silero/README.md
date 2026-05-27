# Silero VAD

## 目录

- [Silero VAD](#silero-vad)
  - [目录](#目录)
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
Silero VAD（Voice Activity Detection，语音活动检测）是一个基于深度学习的实时语音活动检测模型，能够在给定音频流中识别语音片段的起止时间。该模型使用可学习STFT频谱变换+4层Conv1d编码器+LSTMCell解码器的架构，具有参数量小、延迟低、精度高的特点。

本例程对[Silero VAD](https://github.com/snakers4/silero-vad)的预训练JIT模型进行移植，导出为ONNX模型，并编译为BModel使之能在SOPHON BM1684X上进行推理测试。

## 2. 特性
* 支持BM1684X(x86 PCIe, SoC)
* 支持FP16模型编译和推理
* 支持基于sail的Python推理
* 支持基于bmrt的C++推理
* 支持WAV音频文件的语音活动检测
* 支持自定义VAD阈值、语音/静音时长等参数
* 支持语音段保存为独立WAV文件

## 3. 准备模型与数据
建议使用TPU-MLIR编译BModel。原始JIT模型需要先导出为ONNX格式，具体可参考[Silero VAD模型导出](./docs/SileroVAD_Export_Guide.md)。

本例程在`tools`目录下提供了模型导出脚本`export_onnx_clean.py`，通过重建纯PyTorch模型的方式导出无控制流算子的ONNX模型，确保TPU-MLIR能够正常编译。

同时，您需要准备用于测试的WAV音频文件（16kHz采样率）。

通过运行如下脚本，可以下载本例程所需的数据和模型。
```bash
bash scripts/download.sh --all
```

下载的模型包括：
```
./models
├── BM1684X
│   └── silero_vad_bm1684x_f16.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=1
└── onnx
    └── silero_vad_core_clean.onnx           # 导出的onnx模型，3输入(mel, h, c)，3输出(speech_prob, h_new, c_new)
```

测试数据包括：
```
./datasets
└── test.wav                              # 测试用16kHz WAV音频文件
```

## 4. 模型编译
导出的ONNX模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。建议使用TPU-MLIR编译BModel。

模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的"3. 编译ONNX模型"(请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的SDK中获取)。

- 生成FP16 BModel

本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_f16_bmodel.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X**），如：

```bash
./scripts/gen_f16_bmodel.sh
```

执行上述命令会在`models/BM1684X/`下生成`silero_vad_bm1684x_f16.bmodel`文件，即转换好的FP16 BModel。

注意：模型有3个输入（x=[1,576]音频帧, h=[1,128] LSTM隐藏状态, c=[1,128] LSTM细胞状态）和3个输出（out=[1,1]语音概率, h_new=[1,128], c_new=[1,128]），编译时必须指定`--channel_format none`（非图片模型）。

## 5. 例程测试
- [Python例程](./python/README.md)
- [C++例程](./cpp/README.md)

## 6. 精度测试
### 6.1 测试方法

首先，参考[Python例程](python/README.md#23-测试音频)推理要测试的WAV音频文件，生成预测的VAD结果JSON文件。精度验证通过对比TPU推理结果与JIT参考模型的逐帧语音概率差异来完成，可运行以下命令：

```bash
cd python
python3 -c "
import torch
import numpy as np
# 加载JIT参考模型和TPU bmodel，逐帧对比语音概率
# 具体脚本参见 tools/ 目录下的验证工具
"
```

### 6.2 测试结果
在Silero VAD的`torch.jit.load`源码模型上，精度测试结果如下：
|   测试平台  |    测试程序               |              测试模型              | Max Prob Diff | Mean Prob Diff | VAD Segments |
| ---------- | ----------------------- | --------------------------------- | ------------- | -------------- | ------------ |
| SE7-32     | silero_vad.py           | silero_vad_bm1684x_f16.bmodel     |    0.016      |    4.27e-4     |   完全一致     |

> **测试说明**：
> 1. Max Prob Diff为全部帧中语音概率的最大绝对差，Mean Prob Diff为平均绝对差；
> 2. F16量化带来的精度损失极小（<0.02），VAD分段结果与JIT参考模型完全一致；
> 3. SE7系列对应BM1684X；

## 7. 性能测试
### 7.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684X/silero_vad_bm1684x_f16.bmodel
```
测试结果中的`calculate time`就是模型推理的时间（每帧512个采样点 @ 16kHz = 32ms音频）。
测试各个模型的理论推理时间，结果如下：

|   测试平台  |                  测试模型                     | calculate time(ms) |
| ----------- | -------------------------------------------- | ----------------- |
|   SE7-32    | BM1684X/silero_vad_bm1684x_f16.bmodel        |        0.291      |

> **测试说明**：
> 1. 性能测试结果具有一定的波动性；
> 2. `calculate time`为单帧推理时间（对应32ms音频），实时因子=0.291/32≈0.0091；

### 7.2 程序运行性能
参考[Python例程](python/README.md)运行程序，并查看统计的预处理时间、推理时间、后处理时间。

在不同的测试平台上，使用不同的例程、模型进行测试，性能测试结果如下：
|    测试平台  |     测试程序      |             测试模型                     |preprocess(ms)|inference(ms)|postprocess(ms)|real_time_factor|
| ----------- | ---------------- | --------------------------------------- | ------------ | ----------- | ------------- | -------------- |
|   SE7-32    | silero_vad.py    | silero_vad_bm1684x_f16.bmodel           |     0.718    |    0.676    |     0.001     |     0.0211     |
|   SE7-32    | silero_vad_bmrt.soc | silero_vad_bm1684x_f16.bmodel        |     0.003    |    0.210    |     0.000     |     0.0066     |

> **测试说明**：
> 1. 时间单位均为毫秒(ms)，统计的时间均为每帧（512采样点=32ms音频）的处理时间；
> 2. preprocess为预处理时间（上下文拼接），inference为TPU推理时间（含S2D/D2S拷贝），postprocess为VAD后处理时间（概率→语音段）；
> 3. real_time_factor = inference时间 / 音频时长，值越小代表加速比越高；
> 4. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 5. SE7-32的主控处理器为8核CA53@2.3GHz，PCIe上的性能由于处理器的不同可能存在较大差异；

## 8. FAQ
1. **模型编译时出现Shape/Gather算子警告**: 这些算子在ONNX导出时不可避免（来自chunk和reshape操作），但BM1684X支持这些算子，不影响编译结果。

2. **推理结果的语音段与期望不符**: 可调整`--threshold`（语音概率阈值，默认0.5）、`--min_speech_duration_ms`（最小语音时长，默认250ms）、`--min_silence_duration_ms`（最小静音时长，默认100ms）等参数。

其他常见问题请参考[SOPHON-DEMO FAQ](../../docs/FAQ.md)。