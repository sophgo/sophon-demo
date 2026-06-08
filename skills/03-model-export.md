# Skill 3: 模型导出 (PyTorch → ONNX)

## 目标
将 PyTorch 模型导出为 ONNX 格式，为 TPU-MLIR 编译做准备。

## 执行步骤

### 3.1 加载 PyTorch 模型
```python
from original_framework import AutoModel

model = AutoModel(
    model="./model",
    device="cpu",
)
```

### 3.2 提取子模型
目标模型 包含三个需要导出的子模型：
- **子模型1** (编码组件): input_data → submodel1_out, hidden_state, intermediate_values, output_length
- **子模型2** (解码组件): submodel1_out + intermediate_embeds → logits
- **子模型3** (辅助预测组件): submodel1_out → aux_data, submodel3_output_length

### 3.3 导出 ONNX
```python
import torch

# 导出 子模型1
dummy_input = torch.randn(1, 100, 560)  # [batch, seq_len, feat_dim]
dummy_input_len = torch.tensor([100], dtype=torch.int32)
torch.onnx.export(
    model.model.子模型1,
    (dummy_input, dummy_input_len),
    "子模型1.onnx",
    input_names=["input_data", "input_lengths"],
    output_names=["submodel1_out", "hidden_state", "intermediate_values", "output_length"],
    dynamic_axes={
        "input_data": {0: "batch", 1: "T"},
        "input_lengths": {0: "batch"},
        "submodel1_out": {0: "batch", 1: "T"},
        "hidden_state": {0: "batch", 1: "T_plus_1"},
        "intermediate_values": {0: "batch", 1: "T_plus_1"},
        "output_length": {0: "batch"},
    },
    opset_version=13,
)
```

### 3.4 ONNX 验证
```python
import onnx
import onnxruntime

# 验证 ONNX 模型
onnx_model = onnx.load("子模型1.onnx")
onnx.checker.check_model(onnx_model)

# 推理对比
ort_session = onnxruntime.InferenceSession("子模型1.onnx")
ort_outputs = ort_session.run(None, {"input_data": input_np, "input_lengths": length_np})
```

## 关键注意事项

### 动态维度
- `batch` 维度建议设为动态，以支持 batch=1 到 batch=10
- `T` (时间维度) 必须为动态，因不同输入数据长度不同
- 编译 BModel 时指定 `MAX_T` (如 1000)

### 算子兼容性
- 避免使用 TPU-MLIR 不支持的算子
- 优先使用标准 ONNX opset 13
- 复杂算子可能需要拆解或替换

### 输出命名
- 使用有意义的输出名，便于后续推理代码按名称匹配
- 不要依赖输出顺序（不同框架可能重排）

## 检查清单

- [ ] PyTorch 模型能正常加载和推理
- [ ] 子模型成功拆分
- [ ] ONNX 导出无错误
- [ ] ONNX 模型通过 checker 验证
- [ ] ONNX Runtime 推理结果与 PyTorch 一致
- [ ] Dynamic axes 配置正确

## 常见问题

1. **ONNX 导出报错**: 检查是否使用了不支持的操作（如动态控制流）
2. **精度不匹配**: 确认 ONNX Runtime 推理结果与 PyTorch 一致后再编译
3. **模型过大**: 考虑使用 ONNX simplify 或移除冗余节点
