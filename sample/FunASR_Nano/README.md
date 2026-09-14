# FunASR-Nano-2512 语音识别

## 目录

- [1. 简介](#1-简介)
- [2. 特性](#2-特性)
- [3. 准备模型与数据](#3-准备模型与数据)
  - [3.1 自动下载](#31-自动下载)
  - [3.2 手动下载](#32-手动下载)
- [4. 模型编译](#4-模型编译)
  - [4.1 导出 ONNX](#41-导出-onnx)
  - [4.2 编译 BModel](#42-编译-bmodel)
- [5. 例程测试](#5-例程测试)
- [6. 精度测试](#6-精度测试)
  - [6.1 测试方法](#61-测试方法)
  - [6.2 测试结果](#62-测试结果)
- [7. 性能测试](#7-性能测试)
  - [7.1 bmrt_test](#71-bmrt_test)
  - [7.2 程序运行性能](#72-程序运行性能)
- [8. FAQ](#8-faq)

## 1. 简介

FunASR-Nano-2512 是阿里巴巴通义实验室于 2025 年 12 月推出的端到端语音识别大模型，总参数量约 **0.8B**，基于数千万小时真实语音数据训练，支持 **31 种语言**，专为低算力部署场景设计。

**模型架构：**

- **SANM 音频编码器** (SenseVoiceSmall): 70 层，512 维隐藏层，4 头注意力，FSMN (kernel=11) 深度可分离卷积分支
- **Transformer 音频适配器**: 2 层，将 512 维编码器输出映射到 LLM 1024 维空间
- **Qwen3-0.6B LLM 解码器**: 28 层，GQA 16/8，head_dim=128，RMSNorm，~0.6B

**推理工作流（全模型 TPU）：**

```
WAV (16kHz) → FBank (80维, 25ms/10ms, CPU) → LFR (7帧拼接, ×6下采样, CPU)
→ SANM Encoder (70 blocks, TPU) → Audio Adapter (TPU) → audio_embedding
→ Qwen3-0.6B LLM prefill+decode (TPU, sail.EngineLLM, w4bf16) → Text
```

整个模型——SANM 编码器、音频适配器、Qwen3-0.6B LLM 解码器——**全部运行在 TPU 上**，
仅 FBank/LFR 特征提取在 CPU 上运行。LLM 采用 `llm_convert.py` 编译为 w4bf16 bmodel，
通过 `sail.EngineLLM` 加载，音频适配器输出以 VLM 式拼接（embedding-splice）注入 LLM 的
token embedding 缓冲区（参考 [sample/Qwen2-VL](../Qwen2-VL)）。

参考论文：[FunASR-Nano (arXiv:2509.12508)](https://arxiv.org/abs/2509.12508)

## 2. 特性

- 支持 BM1684X SoC (SE7-32)、BM1688 SoC (SE9-16)、BM1684X2 SoC (SE13-64)
- 支持 F16 模型编译和推理（编码器+适配器），LLM 解码器 w4bf16 量化运行在 TPU
- **整个模型（编码器+适配器+Qwen3-0.6B LLM 解码器）全部在 TPU 上运行**，仅 FBank/LFR 在 CPU
- FBank 特征提取 + LFR 预处理（无 CMVN，与 FunASR-Nano 前端一致）
- 31 种语言语音识别（中文、英文、日文等）
- 支持 RTF (Real-Time Factor) 统计，SE7-32 上 RTF ≈ 0.08，SE9-16 上 RTF ≈ 0.16
- 支持热词自定义 (hotword customization)

## 3. 准备模型与数据

### 3.1 自动下载

在 sophon-demo 根目录下运行：

```bash
cd sample/FunASR_Nano/scripts
bash download.sh
```

下载内容包括：

- **BM1684X F16 BModel**: 预编译的编码器、适配器与 Qwen3-0.6B LLM (w4bf16) bmodel + tokenizer 配置
- **BM1688 F16 BModel**: 预编译的编码器、适配器与 LLM bmodel + tokenizer 配置
- **ONNX 模型**: 用于自行编译编码器/适配器
- **测试数据集**: aishell_S0764（96 个 16kHz WAV 样本，与 WeNet 共用）

### 3.2 手动下载

**PyTorch 模型**（首次推理时通过 FunASR AutoModel 自动下载，也可预下载）：

```bash
python3 -c "from funasr import AutoModel; \
    AutoModel(model='FunAudioLLM/Fun-ASR-Nano-2512', trust_remote_code=True)"
```

PyTorch 模型位于 [HuggingFace: FunAudioLLM/Fun-ASR-Nano-2512](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512)。

## 4. 模型编译

### 4.1 导出 ONNX

```bash
cd tools
python3 export_onnx.py
```

导出文件：
- `models/onnx/sanm_encoder.onnx` — SANM 编码器（T=200 grid，约 12s 音频）
- `models/onnx/audio_adapter.onnx` — 音频适配器（T=200 grid）

> 注：ONNX 固定 T=200 帧。如需更长音频，修改 `export_onnx.py` 中 trace 输入的 T 值重新导出。

### 4.2 编译 BModel

TPU-MLIR 环境准备参考 [TPU-MLIR 环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。**要求 TPU-MLIR ≥ v1.28.1**。

#### 4.2.1 编码器 + 适配器 (F16)

```bash
cd scripts
bash gen_fp16bmodel_mlir.sh          # F16（推荐，BM1688）
bash gen_fp16bmodel_mlir.sh bm1684x  # BM1684X 版本
bash gen_fp16bmodel_mlir.sh bm1684x2 # BM1684X2 版本（仅 F16，固件不支持 FP32）
```

编译产物（以 BM1684X 为例）：

```bash
./models/BM1684X
├── funasr_encoder_f16_1b.bmodel       # SANM 编码器, F16 (~431MB)
└── funasr_adapter_f16_1b.bmodel       # 音频适配器, F16 (~31MB)
```

#### 4.2.2 Qwen3-0.6B LLM 解码器 (w4bf16)

LLM 解码器使用 SophonSDK 的 `llm_convert.py` 从 HuggingFace 权重编译为 w4bf16 bmodel。
先从 FunASR-Nano 的 PyTorch 权重中提取标准 `Qwen3ForCausalLM` 权重（去掉 `llm.` 前缀），
连同 `Qwen3-0.6B/` 子目录的 `config.json` / `tokenizer.json` 放入一个 HF 模型目录，
然后在 `llm_convert` 环境（`sophon-llm` 容器）中编译：

```bash
# 提取 LLM 权重（host，torch 可读 Fun-ASR-Nano-2512 model.pt）
python3 tools/extract_llm_weights.py   # 产物 tools/qwen3_0.6b_llm/

# 编译为 w4bf16 bmodel（在 sophon-llm 容器内）
llm_convert -m tools/qwen3_0.6b_llm \
    -c bm1684x --quantize w4bf16 --num_core 1 \
    --max_input_length 256 -s 512 \
    --out_dir models/BM1684X/llm_out
# 产物: qwen3_0.6b_llm_w4bf16_seq512_bm1684x_1dev_static_*.bmodel
# 连同 config/(tokenizer.json 等) 一起放到 python/config/
```

> 注：模型目录名不要同时包含 "qwen" 和 "asr" 两个子串，否则 `llm_convert.py` 会误触发
> `qwen_asr` 导入。命名为 `qwen3_0.6b_llm` 即可。BM1688 平台把 `-c bm1684x` 改为 `-c bm1688`。

最终运行例程所需的模型文件（`models/<CHIP>/`）：

```
models/BM1684X
├── funasr_encoder_f16_1b.bmodel                              # 编码器 F16
├── funasr_adapter_f16_1b.bmodel                              # 适配器 F16
└── qwen3_0.6b_llm_w4bf16_seq512_bm1684x_1dev_static.bmodel   # LLM w4bf16
```
以及 `python/config/`（tokenizer.json、vocab.json、merges.txt 等，随 LLM 编译产物一并放置）。

## 5. 例程测试
- [Python例程](./python/README.md)

## 6. 精度测试

参考 WeNet / Whisper 等例程的精度测试方式，在 aishell 测试子集 `aishell_S0764`（96 个 16kHz
中文样本，与 WeNet 共用数据集）上计算 **CER (Character Error Rate，字错率)**。

### 6.1 测试方法

1. 用 `python/batch_eval.py` 对整个测试集做全模型 TPU 推理，生成 `result.txt`（每行
   `<utt_id> <识别文本>`）：

```bash
python3 batch_eval.py --dataset ../datasets/aishell_S0764 \
    --encoder ../models/BM1684X/funasr_encoder_f16_1b.bmodel \
    --adapter ../models/BM1684X/funasr_adapter_f16_1b.bmodel \
    --llm ../models/BM1684X/qwen3_0.6b_llm_w4bf16_seq512_bm1684x.bmodel \
    --config config/ --dev_id 0 --max_new_tokens 64 --output result.txt
```

2. 用 `tools/eval_aishell.py`（与 WeNet 同一脚本）对照 `ground_truth.txt` 计算 CER：

```bash
python3 ../tools/eval_aishell.py --char=1 --v=0 \
    ../datasets/aishell_S0764/ground_truth.txt result.txt | grep Overall
```

### 6.2 测试结果

在 aishell_S0764（96 条，共 1335 字）上的端到端全模型 TPU 推理结果：

| 测试平台 | 测试程序 | 编码器+适配器 | LLM 解码器 | CER | 平均 RTF |
|----------|----------|--------------|-----------|-----|----------|
| SE7-32 (BM1684X) | batch_eval.py | F16 | w4bf16 | 5.69% | 0.069 |
| SE9-16 (BM1688) | batch_eval.py | F16 | w4bf16 | 3.45% | 0.136 |

> **测试说明**：
> 1. CER = (S+D+I)/N（S=替换 D=删除 I=插入 N=总字数 1335）。SE7: (29+46+1)/1335=5.69%；SE9: (25+19+2)/1335=3.45%；
> 2. `eval_aishell.py --char=1` 按字计算，标点已剥离，不计入错误；
> 3. FunASR-Nano 是 31 语种通用大模型（非 aishell 专精 CTC 模型），3–6% CER 属合理水平；
>    作为参考，WeNet（aishell 专精 CTC）在同一测试集上 CER 约 1.7–2.7%；
> 4. 编码器+适配器要求 TPU-MLIR ≥ v1.28.1（v1.27 编译的 F16 bmodel 输出 NaN）。

## 7. 性能测试

### 7.1 bmrt_test

使用 `bmrt_test` 测试编码器/适配器的理论 TPU 推理性能（`bmrt_test` 位于 `/opt/sophon/libsophon-*/bin/bmrt_test`）：

```bash
bmrt_test --bmodel models/BM1684X/funasr_encoder_f16_1b.bmodel --devid 0
bmrt_test --bmodel models/BM1684X/funasr_adapter_f16_1b.bmodel --devid 0
```

测试结果中的 `calculate time` 即为单次 TPU 推理时间，结果如下：

| 测试平台 | 测试模型 | calculate time(ms) |
|----------|----------|-------------------|
| SE7-32 (BM1684X) | funasr_encoder_f16_1b.bmodel | 28.1 |
| SE7-32 (BM1684X) | funasr_adapter_f16_1b.bmodel | 1.25 |
| SE9-16 (BM1688) | funasr_encoder_f16_1b.bmodel | 106.3 |
| SE9-16 (BM1688) | funasr_adapter_f16_1b.bmodel | 5.8 |
| SE13-64 (BM1684X2) | funasr_encoder_f16_1b.bmodel | 43.4 |
| SE13-64 (BM1684X2) | funasr_adapter_f16_1b.bmodel | 2.4 |

> **测试说明**：
> 1. 性能测试结果具有一定的波动性；
> 2. calculate time 为单次 TPU 推理耗时，不含数据搬运；
> 3. LLM (Qwen3-0.6B w4bf16) 为多子图 bmodel（embedding/block_*/block_cache_*/lm_head），
>    不适用单图 bmrt_test 汇总，其耗时见 §7.2 端到端测试的 llm 项。

### 7.2 程序运行性能（RTF）

参考 [Python例程](python/README.md) 运行 `funasr_nano_infer.py`，例程打印 `encode`（FBank+LFR + 编码器 + 适配器，其中神经网络部分在 TPU）、`llm`（Qwen3-0.6B prefill+decode，全在 TPU）和 **RTF**。

**RTF (Real-Time Factor) = 端到端推理总时间 / 音频时长**，越低越快（<1 表示快于实时）。

| 测试平台 | 编码器+适配器 | LLM | 音频时长(s) | encode(ms) | llm(ms) | total(ms) | RTF |
|----------|--------------|-----|------------|------------|---------|-----------|-----|
| SE7-32 (BM1684X) | F16 | w4bf16 | 5.62 | 140 | 319 | 459 | 0.082 |
| SE7-32 (BM1684X) | F16 | w4bf16 | aishell_S0764 平均 (4.6s/条) | ~75 | ~240 | ~316 | 0.069 |
| SE9-16 (BM1688) | F16 | w4bf16 | 4.20 | 146 | 528 | 675 | 0.160 |
| SE9-16 (BM1688) | F16 | w4bf16 | aishell_S0764 平均 (4.6s/条) | ~90 | ~530 | ~620 | 0.136 |

> **测试说明**：
> 1. `encode` = FBank+LFR 特征提取(CPU) + SANM 编码器(TPU) + 音频适配器(TPU)；
> 2. `llm` = Qwen3-0.6B prefill + 自回归 decode，**全部在 TPU 上运行**（sail.EngineLLM, w4bf16）；
> 3. 整个模型（编码器+适配器+LLM 解码器）全部在 TPU 上，仅 FBank/LFR 在 CPU；
> 4. SE7-32 上 RTF ≈ 0.07–0.08（约实时 12–14 倍），SE9-16 上 RTF ≈ 0.14–0.16（约实时 6–7 倍）；
> 5. SE9-16 (BM1688) 编码器较 SE7-32 (BM1684X) 慢约 3.8 倍（106ms vs 28ms），故整体 RTF 更高；
> 6. 性能测试结果具有一定的波动性，建议多次测试取平均值。

## 8. FAQ

### Q1: 为什么要求 TPU-MLIR ≥ v1.28.1？

v1.27 编译的 F16 bmodel 在 BM1688 上输出 NaN，v1.28.1 修复了此问题。验证版本：

```bash
pip show tpu-mlir | grep Version
```

### Q2: 如何提升精度？

- BM1684X 平台 FP16 编码器精度良好（cos ≈ 0.99999）
- 编码器/适配器可探索 INT8 量化（带校准数据）
- LLM 已为 w4bf16 量化，可尝试 w8bf16 提升精度（bmodel 体积增大）
- 尝试 `--quantize_table` 混精度编译（参考 [Calibration Guide](../../docs/Calibration_Guide.md)）

### Q3: 为什么 ONNX 固定 T=200？

导出时模型内部常数被折叠，导致 grid 固定。需不同 grid 时修改 `tools/export_onnx.py` 中 trace 输入的 T 值重新导出。

### Q4: LLM 解码器如何在 TPU 上运行？

Qwen3-0.6B LLM 解码器通过 SophonSDK 的 `llm_convert.py` 从 HuggingFace 权重编译为 w4bf16 bmodel
（含 embedding / block_* / block_cache_* / lm_head 等子图），运行时用 `sail.EngineLLM` 加载。
音频适配器的输出（audio_embedding）按 VLM 式拼接（参考 [sample/Qwen2-VL](../Qwen2-VL)）注入 LLM
token embedding 缓冲区的语音占位符位置，从而实现整个模型在 TPU 上的端到端推理。

### Q5: 如何编译双核版本？

编码器/适配器编译脚本在 BM1688 上生成 `_2core.bmodel`。LLM 编译时用 `--num_core 2`。
推理时通过 `--dev_id` 指定设备即可。
