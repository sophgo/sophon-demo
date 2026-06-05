# Skill 5: Python 推理验证

## 目标
使用 sophon.sail 加载 BModel 进行 Python 推理，验证模型在 TPU 上能正常输出。

## 执行步骤

### 5.1 加载 BModel
```python
import sophon.sail as sail

encoder = sail.Engine("encoder_fp32_10b.bmodel", dev_id=0, mode=sail.IOMode.SYSIO)
decoder = sail.Engine("decoder_fp32_10b.bmodel", dev_id=0, mode=sail.IOMode.SYSIO)
predictor = sail.Engine("predictor_fp32_10b.bmodel", dev_id=0, mode=sail.IOMode.SYSIO)
```

### 5.2 实现预处理 (CPU)
```python
# 1. 读取音频 (16kHz, mono, float32)
audio = read_audio("test.wav")

# 2. FBANK 特征提取 (80 维 mel 滤波器组)
fbank = kaldi.fbank(waveform, num_mel_bins=80, frame_length=25,
                    frame_shift=10, sample_frequency=16000)

# 3. LFR (Low Frame Rate): m=7, n=6
lfr = apply_lfr(fbank, lfr_m=7, lfr_n=6)  # → [T_lfr, 560]

# 4. CMVN (倒谱均值方差归一化)
lfr = (lfr + cmvn_means) * cmvn_vars
```

### 5.3 逐模型推理
```python
# 编码器
enc_in = {"speech": speech[None,:,:], "speech_lengths": np.array([speech_len])}
enc_out = encoder.process(graph_name, enc_in)
# 输出: enc_out, hidden, alphas, token_num

# CIF (CPU)
pre_embeds, _ = cif(hidden, alphas, threshold=1.0)
pre_embeds = pre_embeds[:, :int(token_num), :]

# 解码器
dec_in = {"enc": enc_out, "enc_len": ..., "pre_acoustic_embeds": pre_embeds, ...}
dec_out = decoder.process(graph_name, dec_in)
# 输出: logits, dec_hidden

# 预测器 V3
pred_in = {"enc": enc_out, "enc_len": ...}
pred_out = predictor.process(graph_name, pred_in)
# 输出: us_alphas, pred_token_num
```

### 5.4 解码 (CPU)
```python
# Greedy 解码
token_ids = np.argmax(logits[0, :N, :], axis=-1)
token_ids = [t for t in token_ids if t not in (SOS, EOS, BLANK)]
tokens = [vocab[tid] for tid in token_ids]
text = "".join(tokens).replace("@@", "").replace(" ", "")

# 时间戳预测
us_alphas = alphas2 * (pre_token_len / pred_token_num)
us_peaks = cif_wo_hidden(us_alphas, threshold=1.0 - 1e-4)
timestamps = ts_prediction_lfr6(us_alphas, us_peaks, tokens)
```

### 5.5 运行测试
```bash
cd python
python3 seaco_paraformer.py \
    --model_dir ../models/BM1684X \
    --input ../model/example/asr_example.wav
```

## 关键验证点

1. **输入 shape 匹配**: 确保输入 tensor shape 和 dtype 与 BModel 的输入描述一致
2. **输出解析**: 确认每个输出 tensor 的含义和 shape
3. **文本解码**: 验证 token_id → token → text 的映射正确
4. **时间戳**: 验证 CIF 峰值检测和时间戳计算的正确性
5. **内存管理**: SoC 模式下使用 zero-copy (mmap)，PCIe 模式使用 d2s 拷贝

## 检查清单

- [ ] BModel 加载成功 (无 BMRT_ASSERT)
- [ ] 预处理输出 shape 正确
- [ ] 编码器输出 shape 符合预期
- [ ] CIF 正确计算 acoustic embeds
- [ ] 解码器输出 logits shape 正确
- [ ] Greedy 解码输出有效文本
- [ ] 时间戳在合理范围内
- [ ] 无 NaN 或 Inf

## 常见问题

1. **BMRT_ASSERT 错误**: bmodel 与 libsophon 版本不兼容
2. **输出全空**: CIF 没有触发任何 token，检查音频和 CMVN
3. **解码乱码**: 检查 tokens.json 与模型是否匹配
4. **内存溢出**: 检查动态 shape 的实际大小是否超过编译时的 max
