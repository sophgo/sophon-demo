# FunASR Nano-2512 模型导出指南

## 目录

- [1. 概述](#1-概述)
- [2. 模型架构](#2-模型架构)
- [3. 环境准备](#3-环境准备)
- [4. ONNX 模型导出](#4-onnx-模型导出)
  - [4.1 全量导出脚本](#41-全量导出脚本)
  - [4.2 SANM 编码器导出](#42-sanm-编码器导出)
  - [4.3 音频适配器导出](#43-音频适配器导出)
  - [4.4 LLM 模块导出](#44-llm-模块导出)
- [5. 算子兼容性说明](#5-算子兼容性说明)
- [6. 验证](#6-验证)
- [7. BModel 编译](#7-bmodel-编译)
  - [7.1 FP32 编译](#71-fp32-编译)
  - [7.2 FP16 编译](#72-fp16-编译)
- [8. 常见问题](#8-常见问题)

## 1. 概述

FunASR-Nano-2512 是一个端到端语音识别大模型（0.8B 参数），由三个子模型串联组成：

| 子模型 | 功能 | 参数量 | 输入 | 输出 |
|--------|------|--------|------|------|
| SANM Encoder | 声学特征编码 | ~0.2B | speech (B,T,560), speech_lengths (B,) | encoder_out (B,T',512), encoder_out_lens (B,) |
| Audio Adapter | 维度映射 | ~few M | encoder_out (B,T',512), encoder_out_lens (B,) | adaptor_out (B,T',1024), adaptor_out_lens (B,) |
| LLM Decoder (Qwen3-0.6B) | 自回归文本解码 | ~0.6B | inputs_embeds (B,N,1024) + 28 blocks | logits (B,N, vocab_size) |

原始模型来自 [FunASR](https://github.com/modelscope/FunASR) 框架。

## 2. 模型架构

```
  Audio (16kHz WAV)
      ↓
  FBank (80-dim, 25ms/10ms)  [CPU: torchaudio.compliance.kaldi]
      ↓
  LFR (7-frame concat + ×6 subsample) → 560-dim
      ↓
  CMVN normalization
      ↓
  [SANM Encoder (70 blocks, 512-dim)]  ← BModel 1
      ↓
  [Audio Adapter (2 blocks)]           ← BModel 2
      ↓
  ChatML Prompt + LLM Embedding (1024-dim)
      ↓
  [LLM Block 0] ← BModel 3-1
  [LLM Block 1] ← BModel 3-2
  ...
  [LLM Block 27] ← BModel 3-28
      ↓
  [LM Head + Greedy] ← BModel 4
      ↓
  Text output
```

### Encoder (SANM) 详细结构

- 70 个 block：1 entry block (560→512) + 49 main blocks + 20 "tp" blocks
- 隐藏维度：512
- 注意力头数：4
- FSMN kernel size：11（深度可分离卷积）
- 输出维度：512

### LLM Decoder (Qwen3-0.6B) 详细结构

- 28 层 Transformer Decoder
- 隐藏维度：1024
- Q 头数：16，KV 头数：8 (GQA)
- Head dim：128
- RoPE θ=1e6
- RMSNorm eps=1e-6
- 词汇量：~151936 (Qwen3 tokenizer)

## 3. 环境准备

```bash
pip install torch torchaudio funasr onnx onnxruntime transformers tiktoken
```

## 4. ONNX 模型导出

### 4.1 全量导出脚本

```bash
cd tools
python3 export_onnx.py \
    --model_path FunAudioLLM/Fun-ASR-Nano-2512 \
    --seq_length 512 \
    --max_audio_frames 3000 \
    --device cpu
```

### 4.2 SANM 编码器导出

编码器输入是 LFR + CMVN 处理后的 FBank 特征：

```python
class SANMEncoderWrapper(torch.nn.Module):
    def forward(self, speech, speech_lengths):
        # speech: (B, T, 560), 560 = 80 (mel_bins) × 7 (lfr_m)
        # speech_lengths: (B,) int32, valid frames count
        encoder_out, encoder_out_lens = self.encoder(speech, speech_lengths)
        return encoder_out, encoder_out_lens

torch.onnx.export(
    encoder_wrapper,
    (dummy_speech, dummy_speech_lengths),
    'sanm_encoder.onnx',
    input_names=['speech', 'speech_lengths'],
    output_names=['encoder_out', 'encoder_out_lens'],
    dynamic_axes={
        'speech': {0: 'batch', 1: 'audio_frames'},
        'encoder_out': {0: 'batch', 1: 'enc_frames'},
    },
    opset_version=14,
)
```

导出后：
```bash
# 简化 ONNX 图
python3 -m onnxsim sanm_encoder.onnx sanm_encoder_sim.onnx
```

### 4.3 音频适配器导出

```python
class AudioAdapterWrapper(torch.nn.Module):
    def forward(self, encoder_out, encoder_out_lens):
        # encoder_out: (B, T', 512), encoder_out_lens: (B,) int32
        adaptor_out, adaptor_out_lens = self.adaptor(encoder_out, encoder_out_lens)
        # adaptor_out: (B, T', 1024)
        return adaptor_out, adaptor_out_lens

torch.onnx.export(
    adapter_wrapper,
    (dummy_enc_out, dummy_enc_out_lens),
    'audio_adapter.onnx',
    input_names=['encoder_out', 'encoder_out_lens'],
    output_names=['adaptor_out', 'adaptor_out_lens'],
    opset_version=14,
)
```

### 4.4 LLM 模块导出

LLM 采用分 block 导出策略（与 `sample/Qwen` 相同）：

**Block（prefill 模式）：**
- 输入：hidden_states (1,512,1024), position_ids (1,512), attention_mask (1,1,512,512)
- 输出：hidden_states (1,512,1024), past_k (1,512,8,128), past_v (1,512,8,128)
- opset_version=15

**Block Cache（decode 模式）：**
- 输入：hidden_states (1,1,1024), position_ids (1,1), attention_mask (1,1,1,513), history_k (1,512,8,128), history_v (1,512,8,128)
- 输出：hidden_states (1,1,1024), past_k (1,1,8,128), past_v (1,1,8,128)

**Embedding** 和 **LM Head** 使用 TorchScript (`.pt`) 格式，在 CPU 端运行：

```python
module = torch.jit.trace(embed.forward, input_ids)
torch.jit.save(module, 'embedding.pt')

module = torch.jit.trace(lm_head.forward, hidden_states)
torch.jit.save(module, 'lm_head.pt')
```

## 5. 算子兼容性说明

| 算子 | 位置 | TPU-MLIR 支持 | 处理方式 |
|------|------|--------------|---------|
| Conv2d | Encoder FSMN 分支 | ✅ 支持 | 直接导出 |
| LayerNorm | Encoder SANM | ✅ 支持 | 直接导出 |
| RMSNorm | LLM Qwen3 | ✅ 支持 | 直接导出 |
| RoPE (cos/sin) | LLM Attention | ✅ 支持 | Pre-compute 后作为常量嵌入 block |
| GQA (Grouped Query Attention) | LLM | ✅ 支持 | opset 15 原生支持 |
| SiLU | LLM FFN | ✅ 支持 | 直接导出 |
| Softmax | Attention | ✅ 支持 | 直接导出 |
| MatMul | Attention, FFN | ✅ 支持 | 直接导出 |
| Where | LLM attention mask | ✅ 支持 | 直接导出 |
| Concat | KV cache | ✅ 支持 | 直接导出 |
| Slice | FSMN, cache_mask | ✅ 支持 | 直接导出 |
| DepthwiseConv1d | FSMN | ⚠️ 需验证 | 可能需替换为 Conv2d+(1,H) 变体 |
| FSMN memory block | Encoder | ⚠️ 需验证 | 用 `Conv1d + padding + NonLinear` 展开 |

> **注意**：如果 TPU-MLIR 编译时报 "unsupported op"，按上表替换后再导出。优先使用 opset 15 以获得更好的 GQA 和 RoPE 支持。

## 6. 验证

导出完成后，使用 onnxruntime 验证 ONNX 输出与 PyTorch 一致：

```bash
python3 export_onnx.py --verify
```

预期输出：
```
Checking sanm_encoder.onnx...
  ONNX model is valid.
  Cosine similarity (encoder): 0.999999
✅ All ONNX models verified successfully!
```

**验证标准**：PyTorch vs ONNX 输出余弦相似度 > 0.9999

## 7. BModel 编译

### 7.1 FP32 编译

```bash
cd scripts
bash gen_fp32bmodel_mlir.sh
```

### 7.2 FP16 编译

```bash
cd scripts
bash gen_fp16bmodel_mlir.sh
```

编译完成后验证：
```bash
for f in models/BM1688/*.bmodel; do
    bmrt_test --bmodel "$f" --dev_id 0
done
```

## 8. 常见问题

### Q1: OP 不支持 (unsupported op)

当编译报错 `unsupported op: xxx` 时：
1. 在 [第 5 节](#5-算子兼容性说明) 的算子兼容表中查找该算子
2. 如果表中有对应替换方案，修改 export_onnx.py 中的模型 wrapper，替换后重新导出
3. 如果不在表中，尝试降低 opset 版本（14→13），或联系 TPU-MLIR 团队

### Q2: 导出时 OOM

FunASR Nano 的 LLM 有 28 层，每个 block 约 85MB ONNX 文件。导出时使用 `--device cpu` 可避免 GPU OOM。

### Q3: Dynamic axes 设置

编码器的输入帧数 T 是动态的（不同音频长度不同），导出时必须设置 `dynamic_axes`：
```python
dynamic_axes={'speech': {1: 'audio_frames'}, 'encoder_out': {1: 'enc_frames'}}
```

编译时用 `--dynamic` 参数保留动态维度，或在 `model_deploy.py` 中指定具体的 `max_input_len`。

### Q4: LLM Block 编译太慢（58 个 bmodel）

28 个 block + 28 个 cache block = 共 56 个 LLM 子模型需要编译。可并行编译加速：
```bash
seq 0 27 | xargs -P 8 -I {} bash -c \
    'model_transform.py ... block_{}.onnx ... && model_deploy.py ...'
```
注意并行数不超过可用内存 / 单个编译所需内存。

### Q5: torchaudio Kaldi FBank 与 Python 实现不一致

必须使用 `torchaudio.compliance.kaldi.fbank` 而非 `torchaudio.transforms.MelSpectrogram`。Kaldi FBank 使用特定参数和归一化方式（`window_type=hamming`, `htk_compat=True`），与标准 mel spectrogram 有差异。
