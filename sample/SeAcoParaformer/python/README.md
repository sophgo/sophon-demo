# Python例程

## 目录

- [Python例程](#python例程)
  - [目录](#目录)
  - [1. 环境准备](#1-环境准备)
    - [1.1 x86/arm/riscv PCIe平台](#11-x86armriscv-pcie平台)
    - [1.2 SoC平台](#12-soc平台)
  - [2. 推理测试](#2-推理测试)
    - [2.1 参数说明](#21-参数说明)
    - [2.2 使用方式](#22-使用方式)
    - [2.3 测试音频](#23-测试音频)

python目录下提供SAIL推理例程：

| 序号 | Python例程              | 说明                    |
| ---- | ----------------------- | ----------------------- |
| 1    | seaco_paraformer.py     | 使用SAIL推理，纯numpy前/后处理 |

## 1. 环境准备

### 1.1 x86/arm/riscv PCIe平台

如果您在x86/arm/riscv平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon、sophon-sail，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)或[riscv-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#6-riscv-pcie平台的开发和运行环境搭建)。

此外您还需要安装其他第三方库：
```bash
pip3 install torch torchaudio soundfile scipy numpy sophon-sail
# torch + torchaudio 用于FBANK特征提取（compliance.kaldi API）
# 若soundfile无法安装，可改用标准库wave
```

### 1.2 SoC平台

如果您使用SoC平台（如SE、SM系列边缘设备），并使用它测试本例程，刷机后在`/opt/sophon/`下已经预装了相应的libsophon运行库包。您还需要交叉编译安装sophon-sail，具体可参考[交叉编译安装sophon-sail](../../../docs/Environment_Install_Guide.md#42-交叉编译安装sophon-sail)。

此外您还需要安装其他第三方库：
```bash
pip3 install torch torchaudio numpy
```

## 2. 推理测试

python例程不需要编译，可以直接运行，PCIe平台和SoC平台的测试参数和运行方式是相同的。

### 2.1 参数说明

```bash
usage: seaco_paraformer.py [--model_dir MODEL_DIR] [--input INPUT] [--dev_id DEV_ID]

--model_dir: 模型目录路径，包含bmodel文件(tokens.json, am.mvn)，默认: ../models/BM1684X；
--input: 输入WAV音频文件路径（16kHz采样率），必填；
--dev_id: TPU设备编号，默认为0；
```

### 2.2 使用方式

模型加载使用SAIL的`sail.Engine`类（`SYSIO`模式，numpy数组输入输出）。

推理流程：
1. **预处理(CPU)**: 音频 → FBANK(80mel) → LFR(m=7,n=6) → CMVN → [1, T, 560]
2. **编码器(TPU)**: [speech, speech_len] → enc_out, hidden, alphas, token_num
3. **CIF(CPU)**: cif(hidden, alphas, threshold=1.0) → pre_acoustic_embeds [1, N, 512]
4. **解码器(TPU)**: [enc_out, enc_len, embeds, token_len] → logits [1, N, vocab]
5. **预测器V3(TPU)**: [enc_out, enc_len] → us_alphas [1, T_up]
6. **后处理(CPU)**: argmax → token_ids → tokens → text

```python
import sophon.sail as sail

# 加载模型
net = sail.Engine(bmodel_path, dev_id, sail.IOMode.SYSIO)
graph_name = net.get_graph_names()[0]

# 推理
inputs = {"speech": speech_array, "speech_lengths": len_array}
outputs = net.process(graph_name, inputs)
```

### 2.3 测试音频

WAV音频测试实例如下（16kHz单声道）：

```bash
cd python
python3 seaco_paraformer.py \
    --model_dir ../models/BM1684X \
    --input ../model/example/asr_example.wav \
    --dev_id 0
```

测试结束后，输出内容包括：
- 识别文本
- 词级别时间戳（[start_ms, end_ms, token]）
- 各阶段耗时（预处理/编码器/CIF/解码器/预测器/解码）
- 实时因子（RTF）

输出示例：
```
[      0][    990]  欢
[    990][   1230]  迎
[   1230][   1290]  大
[   1290][   1530]  家
[   1530][   1610]  来
[   1610][   1830]  到
```

结果会保存为JSON文件至 `./results/` 目录，格式如下：
```json
{
    "audio_file": "../model/example/asr_example.wav",
    "duration_s": 4.52,
    "text": "欢迎大家来到么哒社区进行体验",
    "tokens": ["欢", "迎", "大", "家", "来", "到", "么", "哒", "社", "区", "进", "行", "体", "验"],
    "sentence_info": [
        {"start": 0, "end": 990, "text": "<sil>"},
        {"start": 990, "end": 1230, "text": "欢"},
        {"start": 1230, "end": 1290, "text": "<sil>"},
        {"start": 1290, "end": 1530, "text": "迎"}
    ],
    "wall_time_s": 1.462,
    "rtf": 0.323
}
```