# Skill 3: 模型导出 (PyTorch → ONNX)

## 目标
将 PyTorch 模型导出为 ONNX 格式，为 TPU-MLIR 编译做准备。

## 执行步骤

### 3.1 加载 PyTorch 模型
```python
from funasr import AutoModel

model = AutoModel(
    model="./model",
    device="cpu",
)
```

### 3.2 提取子模型
SeACoParaformer 包含三个需要导出的子模型：
- **Encoder** (SAN-M 编码器): speech → enc_out, hidden, alphas, token_num
- **Decoder** (ParaformerSANM 解码器): enc_out + pre_embeds → logits
- **Predictor** (CifPredictorV3): enc_out → us_alphas, pred_token_num

### 3.3 导出 ONNX
```python
import torch

# 导出 encoder
dummy_speech = torch.randn(1, 100, 560)  # [batch, T, feat_dim]
dummy_speech_len = torch.tensor([100], dtype=torch.int32)
torch.onnx.export(
    model.model.encoder,
    (dummy_speech, dummy_speech_len),
    "encoder.onnx",
    input_names=["speech", "speech_lengths"],
    output_names=["enc_out", "hidden", "alphas", "token_num"],
    dynamic_axes={
        "speech": {0: "batch", 1: "T"},
        "speech_lengths": {0: "batch"},
        "enc_out": {0: "batch", 1: "T"},
        "hidden": {0: "batch", 1: "T_plus_1"},
        "alphas": {0: "batch", 1: "T_plus_1"},
        "token_num": {0: "batch"},
    },
    opset_version=13,
)
```

### 3.4 ONNX 验证
```python
import onnx
import onnxruntime

# 验证 ONNX 模型
onnx_model = onnx.load("encoder.onnx")
onnx.checker.check_model(onnx_model)

# 推理对比
ort_session = onnxruntime.InferenceSession("encoder.onnx")
ort_outputs = ort_session.run(None, {"speech": speech_np, "speech_lengths": len_np})
```

## 关键注意事项

### 动态维度
- `batch` 维度建议设为动态，以支持 batch=1 到 batch=10
- `T` (时间维度) 必须为动态，因不同音频长度不同
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
