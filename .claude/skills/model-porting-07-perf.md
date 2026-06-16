---
name: model-porting-07-perf
description: 性能测试。测试 BModel 理论性能和程序端到端性能，输出延迟和吞吐报告。
---

# 步骤 07：性能测试

## 前置

步骤 06 已完成（精度通过的 BModel 列表已确认）。

## 提示词

```
对精度通过的 BModel 进行性能测试：

1. bmrt_test 理论性能（不含前后处理）：
   bmrt_test --bmodel [path] --dev_id 0
   记录 "calculate time" 作为理论推理时间

2. 程序端到端性能（含全链路）：
   - 运行 Python 推理脚本，--loops 1000
   - 分阶段计时：decode / pre_process / inference / post_process

3. 多 batch 对比（1b/4b）

4. 输出性能测试表：
   - 图像模型：推理延迟(ms/图) + 吞吐(FPS)
   - 语音模型：RTF (Real Time Factor)
   - LLM：tokens/s (prefill + decode)
```

## 预期输出

```markdown
| 芯片 | 精度 | Batch | 理论延迟(ms) | 端到端延迟(ms) | 吞吐(FPS) |
|------|------|-------|------------|-------------|-----------|
| BM1684X | FP16 | 1 | 0.35 | 0.63 | 1587 |
| BM1684X | FP16 | 4 | 0.95 | 1.42 | 2817 |
| BM1684X | INT8 | 4 | 0.62 | 1.08 | 3704 |
```

## 内联知识

| 指标 | 适用场景 | 计算方式 |
|------|---------|---------|
| FPS | 图像模型 | 1000 / avg_latency_ms |
| RTF | 语音模型 | processing_time / audio_duration（<1 表示实时） |
| tokens/s | LLM | total_tokens / total_time |

`bmrt_test` 的 "calculate time" 是 TPU 纯推理时间，端到端延迟在此基础上加上前后处理开销。两者差异大说明前后处理是瓶颈。
