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

**推理工作流：**

```
WAV (16kHz) → FBank (80维, 25ms/10ms) → LFR (7帧拼接, ×6下采样)
→ CMVN → SANM Encoder (70 blocks, TPU) → Audio Adapter (TPU)
→ LLM Decoder (CPU, autoregressive) → Text
```

参考论文：[FunASR-Nano (arXiv:2509.12508)](https://arxiv.org/abs/2509.12508)

## 2. 特性

- 支持 BM1688 SoC
- 支持 F16、FP32 模型编译和推理
- 编码器+适配器运行在 TPU，LLM 解码器运行在 CPU
- FBank 特征提取 + LFR + CMVN 预处理
- 31 种语言语音识别（中文、英文、日文等）
- 支持热词自定义 (hotword customization)

## 3. 准备模型与数据

### 3.1 自动下载

在 sophon-demo 根目录下运行：

```bash
cd sample/FunASR_Nano/scripts
bash download.sh
```

下载内容包括：

- **BM1688 F16 BModel**: 预编译的编码器和适配器（共约 462MB）
- **ONNX 模型**: 用于自行编译
- **测试数据集**: aishell_S0764（96 个 16kHz WAV 样本）

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

```bash
cd scripts
bash gen_fp16bmodel_mlir.sh       # F16（推荐）
bash gen_fp16bmodel_mlir.sh bm1684x   # BM1684X 版本
```

编译产物：

```bash
./models/BM1688
├── funasr_encoder_f16_1b.bmodel       # SANM 编码器, F16 (~431MB)
├── funasr_adapter_f16_1b.bmodel       # 音频适配器, F16 (~31MB)
├── funasr_encoder_f16_1b_2core.bmodel # 双核版本
└── funasr_adapter_f16_1b_2core.bmodel # 双核版本
```

FP32 备选：

```bash
bash gen_fp32bmodel_mlir.sh
```

## 5. 例程测试
- [Python例程](./python/README.md)

## 6. 精度测试

参考 [Python例程](python/README.md) 运行程序，使用 aishell_S0764 子集（96 个样本）对比 PyTorch FP32 基准输出。

在 BM1688 SoC (SE9-16) 上，使用 3.31s 中文测试音频的端到端结果：

| 模型 | 输出 | 正确字数 |
|------|------|----------|
| PyTorch 基准 | `但是由于直销改为经销。` | 11/11 |
| **TPU F16+F16** | `但是由于直销改为直销。` | 10/11 |
| TPU F32+F32 | `但是由于直销改为直` | 8/11 |

> **测试说明**：
> 1. BM1688 F16 精度优于 F32（10 vs 8 字正确），bmodel 体积减半；
> 2. 编译要求 TPU-MLIR ≥ v1.28.1（v1.27 编译的 F16 bmodel 输出 NaN）；
> 3. 精度损失主要来自 70 层 SANM 编码器逐层累积误差，BM1684X 上 FP16 精度完美 (cos≈0.99999)。

## 7. 性能测试

### 7.1 bmrt_test

在 SE9-16 SoC 上，使用 `bmrt_test` 测试模型的理论性能（`bmrt_test` 位于 `/opt/sophon/libsophon-*/bin/bmrt_test`）：

```bash
bmrt_test --bmodel models/BM1688/funasr_encoder_f16_1b.bmodel --devid 0
bmrt_test --bmodel models/BM1688/funasr_adapter_f16_1b.bmodel --devid 0
```

测试结果中的 `calculate time` 即为模型推理时间，结果如下：

| 测试平台 | 测试模型 | calculate time(ms) |
|----------|----------|-------------------|
| SE9-16 | BM1688/funasr_encoder_f16_1b.bmodel | 106.3 |
| SE9-16 | BM1688/funasr_adapter_f16_1b.bmodel | 5.8 |
| SE9-16 | BM1688/funasr_encoder_f32_1b.bmodel | 677.9 |
| SE9-16 | BM1688/funasr_adapter_f32_1b.bmodel | 31.1 |

> **测试说明**：
> 1. 性能测试结果具有一定的波动性；
> 2. calculate time 为单次 TPU 推理耗时，不含数据搬运；
> 3. 单 batch 模型无需折算 batch size。

### 7.2 程序运行性能

参考 [Python例程](python/README.md) 运行程序，测试 3.31s 音频（16kHz, T=55 帧），Python 例程打印的 encode 时间为编码器+适配器 TPU 耗时，llm 为 CPU 解码耗时。性能测试结果如下：

| 测试平台 | 测试模型 | preprocess_time(ms) | encoder_time(ms) | adapter_time(ms) | llm_time(s) |
|----------|----------|---------------------|------------------|------------------|-------------|
| SE9-16 | F16+F16 | 41.7 | 107.2 | 7.4 | 7.1 (x86) |
| SE9-16 | F32+F32 | 41.7 | 678.8 | 32.9 | 6.5 (x86) |

> **测试说明**：
> 1. preprocess_time 为 FBank+LFR 特征提取耗时（单次，含 WAV 解码）；
> 2. encoder/adapter_time 为 10 次循环平均的纯 TPU 推理耗时；
> 3. llm_time 为 Qwen3-0.6B 在 x86 CPU 上的 decode 耗时（SE9 内存不足无法运行）；
> 4. F16 encoder 比 F32 快 6.3x（107.2ms vs 678.8ms）；
> 5. 性能测试结果具有一定的波动性，建议多次测试取平均值。

## 8. FAQ

### Q1: 为什么要求 TPU-MLIR ≥ v1.28.1？

v1.27 编译的 F16 bmodel 在 BM1688 上输出 NaN，v1.28.1 修复了此问题。验证版本：

```bash
pip show tpu-mlir | grep Version
```

### Q2: 如何提升精度？

- BM1684X 平台 FP16 精度完美（cos ≈ 0.99999）
- 探索 INT8 量化（带校准数据）
- 尝试 `--quantize_table` 混精度编译（参考 [Calibration Guide](../../docs/Calibration_Guide.md)）

### Q3: 为什么 ONNX 固定 T=200？

导出时模型内部常数被折叠，导致 grid 固定。需不同 grid 时修改 `tools/export_onnx.py` 中 trace 输入的 T 值重新导出。

### Q4: SE9 上如何避免 OOM？

SE9 可用 RAM 仅 3.3GB，加载完整 PyTorch 模型会 OOM。建议：

1. 仅加载前端（WavFrontend）做特征提取
2. TPU 跑编码器+适配器，保存 embedding 到 .npy
3. 将 embedding 传到 x86 主机跑 LLM 解码

### Q5: 如何编译双核版本？

编译脚本默认生成 `_2core.bmodel`。推理时通过 `--dev_id` 指定设备即可，sophon-sail 会自动选择单/双核模型。
