---
name: model-porting-03-export
description: 将 PyTorch 模型导出为 ONNX。处理算子兼容性替换，验证 ONNX 输出与原始框架一致。
---

# 步骤 03：导出 ONNX

## 前置

步骤 01 已完成（`model_info.json`），步骤 02 环境已就绪。

## 提示词

```
根据 model_info.json，编写 tools/export_onnx.py，将 PyTorch 模型导出为 ONNX：

1. 加载预训练权重
2. model.eval()，构造 dummy_input（shape 与 model_info.json 一致）
3. torch.onnx.export：
   - opset_version=14
   - dynamic_axes 仅包含 batch 维度（后续步骤通过 --dynamic 编译）
   - 输出到 models/onnx/[model_name].onnx
4. 运行 onnx-simplifier 简化模型图
5. onnx.checker.check_model 验证合法性
6. 用 onnxruntime 对比 ONNX 输出与 PyTorch 输出，余弦相似度 > 0.9999

⚠️ 算子兼容性处理（见下方内联知识）：
- 遇到不兼容算子，按表格自动替换后再导出
- 替换后需重新验证 PyTorch 侧推理结果不受影响
```

## 预期输出

- `models/onnx/[model_name].onnx`
- `tools/export_onnx.py`
- 验证日志：PyTorch vs ONNX 输出 diff < 1e-5

## 内联知识：算子兼容表

| 原始算子 | 问题 | 替换方案 |
|---------|------|---------|
| PReLU | TPU-MLIR 不支持 | `LeakyReLU(negative_slope=0.25)` |
| GlobalAveragePool | opset 14+ 格式不兼容 | 降 opset 到 13，或用 `AdaptiveAvgPool2d(1)` |
| LayerNorm | 部分版本不支持 | 手动展开：`(x-mean)/sqrt(var+eps)*weight+bias` |
| torch.fold/unfold | 不支持动态 | 替换为静态 `slice` + `reshape` |
| GELU | TPU-MLIR 支持 | 无需替换 |
| HardSwish | TPU-MLIR 支持 | 无需替换 |
| SiLU/Swish | TPU-MLIR 支持 | 无需替换 |

如果遇到不在表中的算子，先尝试直接导出，在步骤 04 编译时观察是否报 "unsupported op"。

## Debug

| 问题 | 排查方向 |
|------|---------|
| ONNX 导出报错 | 检查模型是否包含动态控制流（if/while），需改写为静态 |
| ONNX 输出与 PyTorch 不一致 | 检查 eval() 模式、BN 层 running stats、dummy_input 值域 |
| 模型过大 | 运行 onnx-simplifier，移除冗余 Identity/Dropout 节点 |
