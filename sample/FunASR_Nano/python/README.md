# Python例程

- [Python例程](#python例程)
  - [1. 环境准备](#1-环境准备)
    - [1.1 x86/arm PCIe平台](#11-x86arm-pcie平台)
    - [1.2 SoC平台](#12-soc平台)
  - [2. 推理测试](#2-推理测试)
    - [2.1 参数说明](#21-参数说明)
    - [2.2 单条音频](#22-单条音频)
    - [2.3 批量 WER 测试](#23-批量-wer-测试)

python目录下提供以下 Python 例程：

| 序号 | Python例程           | 说明                                          |
| ---- | ------------------- | --------------------------------------------- |
| 1    | funasr_nano_infer.py | 端到端全模型 TPU 语音识别（编码器+适配器+LLM） |
| 2    | batch_eval.py        | aishell_S0764 批量推理 + CER 精度测试          |

## 1. 环境准备

### 1.1 x86/arm PCIe平台

如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon、sophon-sail，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。

此外您还需要安装其他第三方库：

```bash
pip3 install torch torchaudio transformers numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 1.2 SoC平台

如果您使用SoC平台（如SE、SM系列边缘设备），并使用它测试本例程，刷机后在`/opt/sophon/`下已经预装了相应的libsophon运行库包。您还需要安装sophon-sail（含 `sail.EngineLLM`），具体可参考[交叉编译安装sophon-sail](../../../docs/Environment_Install_Guide.md#42-交叉编译安装sophon-sail)。

此外您还需要安装其他第三方库：

```bash
pip3 install torch torchaudio transformers numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
# torchaudio 用于 FBank 特征提取，transformers 用于 Qwen3 tokenizer
```

运行前设置环境变量：

```bash
export LD_LIBRARY_PATH=/opt/sophon/libsophon-current/lib:$LD_LIBRARY_PATH
```

## 2. 推理测试

python例程不需要编译，可以直接运行。整个模型（编码器+适配器+Qwen3-0.6B LLM 解码器）全部在 TPU 上运行，仅 FBank/LFR 特征提取在 CPU。

### 2.1 参数说明

```bash
usage: funasr_nano_infer.py [-h] --input INPUT [--encoder ENCODER]
                            [--adapter ADAPTER] [--llm LLM] [--config CONFIG]
                            [--dev_id DEV_ID] [--max_new_tokens MAX_NEW_TOKENS]
--input:           输入 WAV 文件路径 (16kHz 单声道)；
--encoder:         SANM 编码器 bmodel 路径，默认 models/BM1684X/funasr_encoder_f16_1b.bmodel；
--adapter:         音频适配器 bmodel 路径，默认 models/BM1684X/funasr_adapter_f16_1b.bmodel；
--llm:             Qwen3-0.6B LLM w4bf16 bmodel 路径；
--config:          LLM tokenizer 配置目录 (含 tokenizer.json 等)；
--dev_id:          TPU 设备 ID，默认 0；
--max_new_tokens:  LLM 最大生成 token 数，默认 128。
```

### 2.2 单条音频

```bash
python3 funasr_nano_infer.py \
    --input ../datasets/aishell_S0764/BAC009S0764W0121.wav \
    --encoder ../models/BM1684X/funasr_encoder_f16_1b.bmodel \
    --adapter ../models/BM1684X/funasr_adapter_f16_1b.bmodel \
    --llm ../models/BM1684X/qwen3_0.6b_llm_w4bf16_seq512_bm1684x.bmodel \
    --config config/ --dev_id 0
```

测试结束后，会打印识别文本、分阶段耗时和 RTF：

```
Text:       甚至出现交易几乎停滞。
Timings:    encode=82ms  llm=198ms  total=280ms
Audio:      4.20s   fake_tokens=9
RTF:        0.067  (total_time / audio_duration)
```

- `encode`: FBank+LFR(CPU) + 编码器(TPU) + 适配器(TPU)
- `llm`: Qwen3-0.6B prefill+decode，**全部在 TPU 上**（sail.EngineLLM, w4bf16）
- `total`: 端到端总耗时
- `RTF`: 实时率 = total / audio_duration（<1 表示快于实时）

### 2.3 批量 WER 测试

```bash
python3 batch_eval.py --dataset ../datasets/aishell_S0764 \
    --encoder ../models/BM1684X/funasr_encoder_f16_1b.bmodel \
    --adapter ../models/BM1684X/funasr_adapter_f16_1b.bmodel \
    --llm ../models/BM1684X/qwen3_0.6b_llm_w4bf16_seq512_bm1684x.bmodel \
    --config config/ --dev_id 0 --max_new_tokens 64 --output result.txt

# 计算 CER
python3 ../tools/eval_aishell.py --char=1 --v=0 \
    ../datasets/aishell_S0764/ground_truth.txt result.txt | grep Overall
```
