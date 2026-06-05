# Skill 1: 阅读和分析 README

## 目标
从项目 README 中提取关键信息，理解模型架构、数据流和测试要求。

## 执行步骤

### 1.1 识别模型架构
- 阅读 README 中的模型架构说明
- 确认模型类型（编码器-解码器 / 纯编码器 / 非自回归等）
- 确认输入输出规格（采样率、特征维度、token 数）

### 1.2 确认推理 pipeline
```
音频 → 预处理 → 编码器(TPU) → CIF(CPU) → 解码器(TPU) → 预测器(TPU) → 解码(CPU) → 文本
```

### 1.3 确认测试要求
- 精度测试：参考模型（FunASR PyTorch）、测试集（AISHELL-1）、指标（CER/WER）
- 性能测试：bmrt_test 理论性能 + 程序运行 RTF

### 1.4 确认文件结构
- `models/BM1684X/` — BModel 文件
- `python/` — Python SAIL 推理代码
- `cpp/` — C++ bmrt 推理代码
- `scripts/` — 下载/编译脚本

## 检查清单

- [ ] 理解模型架构（encoder/decoder/predictor 结构）
- [ ] 确认输入格式（16kHz 单声道 WAV, FBANK 80维, LFR m=7 n=6）
- [ ] 确认输出格式（文本 + 词级别时间戳）
- [ ] 确认依赖（libsophon, sophon-sail, torch, torchaudio, numpy）
- [ ] 理解精度/性能测试方法
- [ ] 确认目标平台（BM1684X, x86 PCIe / SE7-32 SoC）

## 示例: SeACoParaformer

```
输入: 16kHz WAV → FBANK(80) → LFR(7,6) → CMVN → [1, T, 560]
编码器: [1, T, 560] → enc_out[1, T, 512], hidden[1, T+1, 512], alphas[1, T+1], token_num[1]
CIF: hidden + alphas → pre_acoustic_embeds[1, N, 512]
解码器: [enc_out, pre_embeds] → logits[1, N, 8404]
预测器: enc_out → us_alphas[1, T*3], pred_token_num[1]
解码: argmax(logits) → token_ids → tokens → text
输出: text + [start_ms, end_ms, token] 时间戳列表
```
