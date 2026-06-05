# Skill 7: 性能测试

## 目标
测试 BModel 在 TPU 上的理论性能和程序端到端性能（RTF）。

## 执行步骤

### 7.1 bmrt_test 理论性能
测试每个 BModel 的纯推理性能（不含前后处理）：

```bash
# 编码器性能
bmrt_test --bmodel models/BM1684X/encoder_fp32_10b.bmodel --devid 0

# 解码器性能
bmrt_test --bmodel models/BM1684X/decoder_fp32_10b.bmodel --devid 0

# 预测器性能
bmrt_test --bmodel models/BM1684X/predictor_fp32_10b.bmodel --devid 0
```

bmrt_test 输出关注指标:
- `calculate time`: 纯推理时间
- `latency`: 延迟 (ms)

### 7.2 程序端到端性能

```bash
# Python
cd python
python3 seaco_paraformer.py \
    --model_dir ../models/BM1684X \
    --input ../model/example/asr_example.wav

# C++ (PCIE)
cd cpp/seaco_paraformer_bmrt
./seaco_paraformer_bmrt.pcie \
    --model_dir ../../models/BM1684X \
    --input ../../model/example/asr_example.wav

# C++ (SoC)
cd /data/seaco_paraformer/bmrt
./seaco_paraformer_bmrt.soc \
    --model_dir /data/seaco_paraformer/models/BM1684X \
    --input /data/seaco_paraformer/audio/asr_example.wav
```

### 7.3 分解各阶段耗时
```
preprocess (CPU):  FBANK + LFR + CMVN
encoder (TPU):     SAN-M 编码器推理
cif (CPU):         CIF 峰值检测
decoder (TPU):     SAN-M 解码器推理
predictor (TPU):   CifPredictorV3 推理
decode (CPU):      argmax + token 解码
─────────────────────────────────────
total:             总耗时
RTF:               total / audio_duration
```

### 7.4 多次测试取平均
```bash
# 运行 5 次取平均，减少波动
for i in 1 2 3 4 5; do
    ./seaco_paraformer_bmrt.soc \
        --model_dir ... --input ... 2>&1 | \
        grep -E "preprocess|encoder|decoder|total|RTF"
done
```

## 性能指标

| 指标 | 说明 | 目标 |
|------|------|------|
| RTF | Real Time Factor (总时间/音频时长) | < 1.0 实时 |
| encoder 时间 | TPU encoder 推理 | < 0.2s (4.5s 音频) |
| decoder 时间 | TPU decoder 推理 | < 0.1s (4.5s 音频) |
| 预处理时间 | CPU FBANK+LFR+CMVN | 取决于 CPU 性能 |

## 性能对比表模板

| 测试平台 | 测试程序 | preprocess(s) | encoder(s) | decoder(s) | total(s) | RTF |
|----------|----------|--------------|-----------|-----------|---------|-----|
| x86 PCIE | Python | ? | ? | ? | ? | ? |
| x86 PCIE | C++ | ? | ? | ? | ? | ? |
| SE7-32 | Python | ? | ? | ? | ? | ? |
| SE7-32 | C++ | ? | ? | ? | ? | ? |

## 检查清单

- [ ] bmrt_test 对所有 BModel 完成
- [ ] Python 程序端到端完成 (x86 PCIe)
- [ ] Python 程序端到端完成 (SoC)
- [ ] C++ 程序端到端完成 (x86 PCIe)
- [ ] C++ 程序端到端完成 (SoC)
- [ ] 5 次测试取平均
- [ ] 各阶段耗时分析合理
- [ ] RTF 满足实时要求

## 性能优化建议

1. **预处理加速**: 使用多线程或 GPU 做 FBANK，或使用 C++ 实现
2. **批量推理**: 利用 batch=10 的 BModel，一次处理多条音频
3. **流水线**: 预处理和 TPU 推理并行（当前音频推理时预处理下一段）
4. **INT8 量化**: 从 FP32 转为 INT8，可能获得 2x 加速（需验证精度）
5. **SoC zero-copy**: 在 SoC 上使用 bm_mem_mmap_device_mem 避免数据拷贝

## 示例结果 (SeACoParaformer)

| 测试平台 | 测试程序 | preprocess(s) | encoder(s) | decoder(s) | total(s) | RTF |
|----------|----------|--------------|-----------|---------|---------|-----|
| x86 PCIE | Python (sail) | 1.307 | 0.106 | 0.031 | 1.462 | 0.323 |
| SE7-32 | Python (sail) | 4.046 | 0.113 | 0.051 | 4.234 | 0.937 |
| SE7-32 | C++ (bmrt) | 5.338 | 0.136 | 0.058 | 5.562 | 1.230 |
