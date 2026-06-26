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

### CLIP / Transformer 模型导出（实战经验，YOLO_world_v2）

含 `nn.MultiheadAttention` 的模型（CLIP 文本编码器、Transformer 等）导出有坑：

| 场景 | 问题 | 解决 |
|------|------|------|
| torch ≥ 2.0 导出 MHA | `aten::scaled_dot_product_attention` 无 onnx symbolic，导不出 | **用 torch 1.13 venv 导出**（MHA 走标准算子） |
| 自定义 MHA 替换 SDPA | onnx 数值可能正确（cos=1.0 vs 原模型），但 **TPU-MLIR 会编译成退化图**（bmodel 输出与 onnx 不一致） | 不要用自定义 MHA 替换；必须用 torch 1.13 让真实 MHA 导出 |
| CLIP 文本编码器 | 是**因果掩码**（`build_attention_mask` 上三角 -inf），自定义注意力漏 attn_mask 会让嵌入 cos≈0.30 | 自定义注意力必须应用 attn_mask；或直接用 torch 1.13 真实 MHA（自带 mask） |
| 多输出 onnx | 新版 ultralytics（8.4.75）Detect 头会导出 6 个输出（特征图+合并输出） | 用 onnx 剪枝为单输出（`del graph.output[:]` 只留需要的），再 onnxsim，编译更干净 |

> torch 1.13 venv 示例：`python -m venv /opt/clip_export_venv && pip install torch==1.13.1 onnx onnxsim onnxruntime ftfy regex "numpy<2"`，CLIP 源码 `git clone --depth 1 https://github.com/openai/CLIP.git` 后 `PYTHONPATH=/opt/clip_src venv/bin/python export.py`。主检测模型仍用容器默认 torch 2.0 + 框架包导出。

### 验证文本嵌入子模型

开放词汇模型（YOLO-World 等）的文本编码器产出的嵌入必须与主模型训练时一致。导出后用 onnxruntime 算嵌入，与原始框架 `encode_text`（L2 归一化后）比**余弦相似度 > 0.9999**。注意 ultralytics 的 `model.txt_feats` == OpenAI CLIP `encode_text` + L2 归一化（cos=1.0），可作为参考。

## Debug

| 问题 | 排查方向 |
|------|---------|
| ONNX 导出报错 | 检查模型是否包含动态控制流（if/while），需改写为静态 |
| ONNX 输出与 PyTorch 不一致 | 检查 eval() 模式、BN 层 running stats、dummy_input 值域 |
| 模型过大 | 运行 onnx-simplifier，移除冗余 Identity/Dropout 节点 |
| `aten::scaled_dot_product_attention` / `aten::unflatten` 不支持 | torch 2.0+ 的 MHA 走 SDPA，降 torch 到 1.13 导出（见上方 CLIP/Transformer 节），勿用自定义 MHA 替换（TPU-MLIR 会误编译） |
| onnx 数值对但 bmodel 输出不一致 | 多见于自定义替换的算子（如自定义 MHA）。TPU-MLIR 编译退化→回到 torch 1.13 用真实算子导出 |
