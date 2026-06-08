# Skill 7: 性能测试

## 目标
测试 BModel 在 TPU 上的理论性能和程序端到端性能（延迟/吞吐指标）。

## 执行步骤

### 7.1 bmrt_test 理论性能
测试每个 BModel 的纯推理性能（不含前后处理）：

```bash
# 编码器性能
bmrt_test --bmodel models/BM1684X/submodel1_fp32.bmodel --devid 0

# 解码器性能
bmrt_test --bmodel models/BM1684X/submodel2_fp32.bmodel --devid 0

# 预测器性能
bmrt_test --bmodel models/BM1684X/submodel3_fp32.bmodel --devid 0
```

bmrt_test 输出关注指标:
- `calculate time`: 纯推理时间
- `latency`: 延迟 (ms)

### 7.2 程序端到端性能

```bash
# Python
cd python
python3 model_inference.py \
    --model_dir ../models/BM1684X \
    --input ../data/test_input

# C++ (PCIE)
cd cpp/model_inference_bmrt
./model_inference_bmrt.pcie \
    --model_dir ../../models/BM1684X \
    --input ../../data/test_input

# C++ (SoC)
cd /data/model_inference/bmrt
./model_inference_bmrt.soc \
    --model_dir /data/model_inference/models/BM1684X \
    --input /data/model_inference/input_data/test_input
```

### 7.3 分解各阶段耗时
```
preprocess (CPU):  特征提取 + 归一化
子模型1 (TPU):     模型编码组件推理
intermediate_process (CPU):         中间处理峰值检测
子模型2 (TPU):     模型解码组件推理
子模型3 (TPU):   模型辅助预测组件 推理
decode (CPU):      argmax + 输出解码
─────────────────────────────────────
total:             总耗时
延迟/吞吐指标:               总耗时 / 输入数据时长
```

### 7.4 多次测试取平均
```bash
# 运行 5 次取平均，减少波动
for i in 1 2 3 4 5; do
    ./model_inference_bmrt.soc \
        --model_dir ... --input ... 2>&1 | \
        grep -E "preprocess|子模型1|子模型2|total|延迟/吞吐指标"
done
```

## 性能指标

| 指标 | 说明 | 目标 |
|------|------|------|
| 延迟/吞吐指标 | 延迟/吞吐指标（总耗时/输入数据时长） | 满足业务需求 |
| 子模型1 推理时间 | TPU 子模型1 推理 | 根据模型确定 |
| 子模型2 推理时间 | TPU 子模型2 推理 | 根据模型确定 |
| 预处理时间 | CPU 特征提取+降采样+归一化 | 取决于 CPU 性能 |

## 性能对比表模板

| 测试平台 | 测试程序 | preprocess(s) | 子模型1(s) | 子模型2(s) | total(s) | 延迟/吞吐指标 |
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
- [ ] 延迟/吞吐指标 满足业务需求

## 性能优化建议

1. **预处理加速**: 使用多线程或 加速器 做 特征提取，或使用 C++ 实现
2. **批量推理**: 利用 batch=10 的 BModel，一次处理多条输入数据
3. **流水线**: 预处理和 TPU 推理并行（当前输入数据推理时预处理下一段）
4. **INT8 量化**: 从 FP32 转为 INT8，可能获得 2x 加速（需验证精度）
5. **SoC zero-copy**: 在 SoC 上使用 bm_mem_mmap_device_mem 避免数据拷贝

## 示例结果 (目标模型)

| 测试平台 | 测试程序 | preprocess(s) | 子模型1(s) | 子模型2(s) | total(s) | 延迟/吞吐指标 |
|----------|----------|--------------|-----------|---------|---------|-----|
| x86 PCIE | Python (sail) | 1.307 | 0.106 | 0.031 | 1.462 | 0.323 |
| SE7-32 | Python (sail) | 4.046 | 0.113 | 0.051 | 4.234 | 0.937 |
| SE7-32 | C++ (bmrt) | 5.338 | 0.136 | 0.058 | 5.562 | 1.230 |
