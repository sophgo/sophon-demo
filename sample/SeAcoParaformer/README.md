# SeACoParaformer 语音识别

## 目录

- [SeACoParaformer 语音识别](#seacoparaformer-语音识别)
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
SeACoParaformer（Semantic-Augmented Contextual Paraformer）是一种非自回归（non-autoregressive）语音识别模型，支持灵活有效的热词（hotword）自定义能力。该模型基于[SAN-M](https://arxiv.org/abs/2210.08424)编码器架构，使用CIF（Continuous Integrate-and-Fire）机制进行长度预测，在保证高精度的同时大幅提升了推理速度。

本例程将SeACoParaformer的预训练模型导出为bmodel，使其能在SOPHON BM1684X上进行推理。模型支持基于sail的纯Python推理，无需FunASR框架依赖。

参考论文：[SeACo-Paraformer: A Non-Autoregressive ASR System with Flexible and Effective Hotword Customization Ability](https://arxiv.org/abs/2308.03266)

## 2. 特性
* 支持BM1684X(x86 PCIe, SoC)
* 支持FP32模型编译和推理
* 支持基于sophon.sail的Python推理
* 支持WAV音频文件的中文语音识别
* 支持热词自定义（hotword customization）
* 支持词级别时间戳输出
* 同时支持VAD（语音端点检测）、PUNC（标点恢复）、SPK（说话人识别）等辅助模型

## 3. 准备模型与数据

建议使用TPU-MLIR编译BModel。原始PyTorch模型需要先导出为ONNX格式，具体可参考[SeACoParaformer模型导出](./docs/Export_Guide.md)。

通过运行如下脚本，可以下载本例程所需的模型和配置文件。

```bash
bash scripts/download.sh
```

下载的模型包括：
```
./models/BM1684X
├── encoder_fp32_10b.bmodel          # 编码器，batch=10, FP32
├── decoder_fp32_10b.bmodel          # 解码器，batch=10, FP32
├── predictor_fp32_10b.bmodel        # CIF预测器V3，batch=10, FP32
├── config.yaml                      # 模型配置文件
├── tokens.json                      # 词表（8404 tokens）
├── am.mvn                           # CMVN均值/方差文件
└── seg_dict                         # 分词词典
```

## 4. 模型编译

导出的ONNX模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。建议使用TPU-MLIR编译BModel。

模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的"3. 编译ONNX模型"(请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的SDK中获取)。

## 5. 例程测试

### Python例程

运行前需要安装依赖：
```bash
pip install sophon-sail numpy torch torchaudio soundfile scipy
```

运行推理：
```bash
cd python
python3 seaco_paraformer.py --model_dir ../models/BM1684X --input ../model/example/asr_example.wav
```

参数说明：
- `--model_dir`: 模型目录路径，包含bmodel文件和配置文件
- `--input`: 输入WAV文件路径（16kHz单声道）
- `--dev_id`: TPU设备ID（默认0）

输出示例：
```
[   1320][   1800]  你
[   1800][   2400]  好
[   2400][   3100]  欢
[   3100][   3700]  迎
[   3700][   4600]  使
[   4600][   5300]  用
```

推理结果会保存到 `./results/` 目录下的JSON文件中。

## 6. 精度测试

### 6.1 测试方法

与FunASR框架的PyTorch推理结果进行对比，测试WER（词错误率）和CER（字符错误率）。

### 6.2 测试结果

在SeACoParaformer的FunASR源码模型上，精度测试结果如下：
|   测试平台  |    测试程序               |              测试模型              | CER | WER |
| ---------- | ----------------------- | --------------------------------- | --- | --- |
| x86 PCIE   | eval_accuracy.py         | encoder/decoder/predictor FP32     | 0.00% | 0.00% |
| SE7-32     | seaco_paraformer.py     | encoder/decoder/predictor FP32     | TBD | TBD |

> **测试说明**：
> 1. CER/WER使用FunASR PyTorch模型在CPU上的推理结果作为参考（ground truth）；
> 2. x86 PCIE测试使用4个WAV音频样本（含短句和长句），TPU bmodel识别结果与PyTorch参考完全一致，CER=0.00%, WER=0.00%；
> 3. 完整测试集（如AISHELL-1 test，7176条音频）的评估可在下载AISHELL-1数据集后运行 `python3 eval_accuracy.py --model_dir ../models/BM1684X --test_manifest <manifest_path> --audio_base <aishell_path>` 进行；
> 4. FP32精度应与PyTorch参考模型完全一致；

## 7. 性能测试

### 7.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684X/encoder_fp32_10b.bmodel
bmrt_test --bmodel models/BM1684X/decoder_fp32_10b.bmodel
bmrt_test --bmodel models/BM1684X/predictor_fp32_10b.bmodel
```

### 7.2 程序运行性能

|    测试平台  |     测试程序               |             测试模型                     | preprocess(s) | encoder(s) | decoder(s) | total(s) | RTF   |
| ----------- | ------------------------- | --------------------------------------- | ------------- | ---------- | ---------- | -------- | ----- |
|   SE7-32    | seaco_paraformer.py       | encoder/decoder/predictor FP32        |  4.046       |  0.113     |  0.051     |  4.234   | 0.937 |
|   SE7-32    | seaco_paraformer_bmrt.soc | encoder/decoder/predictor FP32        |  5.338       |  0.136     |  0.058     |  5.562   | 1.230 |
|   x86 PCIE  | seaco_paraformer.py       | encoder/decoder/predictor FP32        |  1.307       |  0.106     |  0.031     |  1.462   | 0.323 |

> **测试说明**：
> 1. RTF = total_time / audio_duration，值越小代表加速比越高；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. SE7-32平台预处理（FBANK特征提取）耗时较长，是ARM AARCH64 CPU性能限制所致，编码器/解码器推理均在TPU上高效完成；x86 PCIE平台CPU性能更强，预处理速度明显更快；
> 4. C++预处理使用纯Armadillo实现，耗时较Python（torchaudio后端）更长，但TPU推理部分性能相当；
> 5. 测试音频：4.52秒，16kHz单声道WAV；

## 8. FAQ

1. **模型加载失败**: 请确认bmodel的SDK版本与当前系统的libsophon版本一致，bmodel不兼容会导致`BMRT_ASSERT`错误。

2. **推理结果为空**: 可能是CIF模型没有触发任何token，检查音频是否包含有效语音，以及CMVN文件是否正确加载。

3. **中文输出乱码**: 请确认`tokens.json`文件与模型匹配，且编码为UTF-8。

其他常见问题请参考[SOPHON-DEMO FAQ](../../docs/FAQ.md)。