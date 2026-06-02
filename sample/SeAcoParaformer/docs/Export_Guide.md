# SeACoParaformer 模型导出指南

## 目录

- [1. 概述](#1-概述)
- [2. 模型架构](#2-模型架构)
- [3. 环境准备](#3-环境准备)
- [4. ONNX 模型导出](#4-onnx-模型导出)
  - [4.1 编码器导出](#41-编码器导出)
  - [4.2 解码器导出](#42-解码器导出)
  - [4.3 预测器导出](#43-预测器导出)
- [5. BModel 编译](#5-bmodel-编译)
  - [5.1 编码器编译](#51-编码器编译)
  - [5.2 解码器编译](#52-解码器编译)
  - [5.3 预测器编译](#53-预测器编译)
- [6. 验证](#6-验证)
- [7. 常见问题](#7-常见问题)

## 1. 概述

SeACoParaformer 是一个非自回归端到端语音识别模型，包含三个需要编译为 bmodel 的子模型：

| 子模型 | 功能 | 输入 | 输出 |
| ------ | ---- | ---- | ---- |
| Encoder | 声学特征编码 | speech (B,T,560), speech_lengths (B,) | enc_out (B,T,512), hidden (B,T+1,512), alphas (B,T+1), token_num (B,) |
| Decoder | 序列解码 | enc_out, enc_lens, pre_embeds, ys_lens | logits (B,N,V), hidden (B,N,512) |
| Predictor V3 | 时间戳预测 | enc_out, enc_lens | us_alphas (B,T×3), token_num (B,) |

原始模型来自 [FunASR](https://github.com/alibaba-damo-academy/FunASR) 框架。

## 2. 模型架构

```
  Audio → FBANK → LFR → CMVN → [Encoder (SAN-M)] → {enc_out, hidden, alphas, token_num}
                                    ↓
                            [CIF (CPU)] → pre_acoustic_embeds
                                    ↓
  [Decoder (ParaformerSANM)] → logits → [SeACo Decoder] + hotword → token IDs → text
                                    ↓
  [Predictor V3] → us_alphas → [CIF CPU] → us_cif_peak → timestamps
```

- **Encoder**: SAN-M结构（50层，512维隐藏层，2048维FFN，8头注意力）
- **Decoder**: ParaformerSANMDecoder（16层，512维）
- **CIF Predictor V3**: Conv1d+BLSTM+ConvTranspose1d（3×上采样），用于生成词级别时间戳
- **SeACo Decoder**: 4层Transformer，用于热词定制化

## 3. 环境准备

```bash
pip install torch funasr onnx onnxruntime
```

参考仓库 `FunASR-bmodel/funasr/models/seaco_paraformer/export_meta.py` 中的导出代码。

## 4. ONNX 模型导出

### 4.1 编码器导出

Encoder 是第一个需要导出的子模型。关键步骤：

```python
from funasr.models.seaco_paraformer.export_meta import export_rebuild_model
from funasr import AutoModel

# 加载模型
model = AutoModel(model="speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404",
                  device="cpu", disable_update=True)
model = model.model

# 重建为导出模型
backbone_model, embedder_model = export_rebuild_model(
    model,
    encoder="SANMEncoder",
    predictor="CifPredictorV3",
    decoder="ParaformerSANMDecoder",
    seaco_decoder="ParaformerSANMDecoder",
    max_seq_len=512,
)

# 导出为ONNX（包含encoder+predictor+decoder+seaco_decoder）
torch.onnx.export(backbone_model, ...)
```

Encoder单独导出需要拆分`export_backbone_forward`方法，单独导出encoder部分。

### 4.2 解码器导出

解码器接受 encoder 输出和 CIF 结果作为输入。

```python
# 解码器输入
inputs = (encoder_out,          # (B, T, 512)
          encoder_out_lens,     # (B,) int32
          pre_acoustic_embeds,  # (B, N, 512)
          ys_pad_lens)          # (B,) int32

# 解码器输出
outputs = (logits,     # (B, N, vocab_size)
           hidden)     # (B, N, 512)
```

### 4.3 预测器导出

CifPredictorV3 用于时间戳预测。

```python
# 预测器输入
inputs = (encoder_out,      # (B, T, 512)
          encoder_out_lens) # (B,) int32

# 预测器输出
outputs = (us_alphas,    # (B, T*3) 上采样alpha值
           token_num)    # (B,) alpha总和
```

## 5. BModel 编译

使用 TPU-MLIR 将 ONNX 模型编译为 bmodel。

### 5.1 编码器编译

```bash
# 编译FP32 bmodel
model_transform.py \
    --model_name seaco_encoder2 \
    --model_def encoder.onnx \
    --input_shapes [[10,1000,560],[10]] \
    --input_types F32,I32 \
    --mlir encoder.mlir

model_deploy.py \
    --mlir encoder.mlir \
    --quantize F32 \
    --chip bm1684x \
    --model encoder_fp32_10b.bmodel
```

### 5.2 解码器编译

```bash
model_transform.py \
    --model_name seaco_decoder \
    --model_def decoder.onnx \
    --input_shapes [[10,1000,512],[10],[10,50,512],[10]] \
    --input_types F32,I32,F32,I32 \
    --mlir decoder.mlir

model_deploy.py \
    --mlir decoder.mlir \
    --quantize F32 \
    --chip bm1684x \
    --model decoder_fp32_10b.bmodel
```

### 5.3 预测器编译

```bash
model_transform.py \
    --model_name seaco_predictor \
    --model_def predictor.onnx \
    --input_shapes [[10,1000,512],[10]] \
    --input_types F32,I32 \
    --mlir predictor.mlir

model_deploy.py \
    --mlir predictor.mlir \
    --quantize F32 \
    --chip bm1684x \
    --model predictor_fp32_10b.bmodel
```

## 6. 验证

编译完成后，使用 `bmrt_test` 验证各模型：

```bash
bmrt_test --bmodel encoder_fp32_10b.bmodel
bmrt_test --bmodel decoder_fp32_10b.bmodel
bmrt_test --bmodel predictor_fp32_10b.bmodel
```

## 7. 常见问题

1. **ONNX导出时opset版本**: 建议使用opset_version=14或更高，确保支持所有算子。

2. **动态维度**: 如果需要可变batch size，在导出和编译时声明动态维度。

3. **log_softmax位置**: 原始PyTorch模型在decoder中做log_softmax；如果bmodel不包含此操作，推理代码需要额外处理。当前bmodel直接输出logits（未归一化），argmax解码不受影响。

4. **Hotword embedding**: 热词模型（BiasEncoder）包含LSTM，可能需要在编译时特殊处理。

5. **SDK版本兼容性**: 编译的bmodel必须与运行时的libsophon版本匹配。不匹配会导致`BMRT_ASSERT: _kernel_modules`错误。
