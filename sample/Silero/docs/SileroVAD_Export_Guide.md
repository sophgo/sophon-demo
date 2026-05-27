# Silero VAD模型导出

## 1. 准备工作
Silero VAD模型导出需要在Pytorch环境下进行，需提前安装好Pytorch环境及相关依赖，并从[Silero VAD GitHub仓库](https://github.com/snakers4/silero-vad)获取预训练JIT模型。

```bash
# 安装依赖
pip3 install torch onnx onnxruntime soundfile numpy
```

确保JIT模型文件`silero_vad.jit`在`upstream/src/silero_vad/data/`目录下。

## 2. 主要步骤

### 2.1 模型架构说明

Silero VAD的核心模型（`_model`，16kHz版本）由以下部分组成：

| 模块 | 结构 | 参数 |
| ---- | ---- | ---- |
| STFT | ReflectionPad1d(0,64) + Conv1d(1,258,256,stride=128) → 幅度谱 | n_fft=256, hop=128 |
| Encoder | 4 × (Conv1d + ReLU) | conv1: 129→128(s1), conv2: 128→64(s2), conv3: 64→64(s2), conv4: 64→128(s1) |
| Decoder | LSTMCell(128,128) + Dropout(0.1) + ReLU + Conv1d(128,1,1) + Sigmoid | hidden_size=128 |

模型输入：
- `x`: [batch, 576] — 64样本历史上下文 + 512样本当前帧 @ 16kHz
- `h`: [batch, 128] — LSTM隐藏状态（初始化为0）
- `c`: [batch, 128] — LSTM细胞状态（初始化为0）

模型输出：
- `out`: [batch, 1] — 当前帧的语音概率 (0~1)
- `h_new`: [batch, 128] — 更新后的LSTM隐藏状态
- `c_new`: [batch, 128] — 更新后的LSTM细胞状态

### 2.2 导出ONNX模型

原始JIT模型存在以下问题，会导致ONNX导出失败或TPU-MLIR编译失败：

1. **控制流算子**: JIT wrapper包含`If`/`Size`/`Shape`等控制流算子，BM1684X TPU不支持
2. **`unsafe_chunk`**: `nn.LSTMCell`内部的chunk操作在ONNX导出时报错

因此，本例程通过重建纯PyTorch模型（`tools/export_onnx_clean.py`）来绕过这些问题：

- `SileroSTFT`: 使用`ReflectionPad1d((0, 64))` + `Conv1d`实现STFT（注意填充只在右侧64样本）
- `SileroEncoder`: 4层Conv1d逐层定义，避免原模型中的自定义Block类
- `SileroDecoder`: 手动实现LSTM门控逻辑（`w_ih`, `w_hh`, `b_ih`, `b_hh`直接做矩阵乘法+chunk），替代`nn.LSTMCell`
- 状态传播: 返回**原始LSTM隐藏状态**（而非经过Dropout/ReLU后的值），确保逐帧推理时状态正确

执行导出：
```bash
cd tools
python3 export_onnx_clean.py \
    --jit ../upstream/src/silero_vad/data/silero_vad.jit \
    --output silero_vad_core_clean.onnx \
    --opset 16
```

导出完成后会进行以下验证：
- ONNX模型结构检查
- 控制流算子检测（若无If/Size/Equal/Not等算子，则适合TPU-MLIR编译）
- PyTorch vs ONNX Runtime数值精度对比（最大误差应 < 1e-7）

### 2.3 验证模型精度

导出脚本会自动对比纯PyTorch重建模型与JIT源模型的输出：
```
Max diff (out):    0.000000e+00
Max diff (h):      0.000000e+00
Max diff (c):      0.000000e+00
```

确保所有差异在浮点精度范围内（< 1e-7），否则编译出的BModel推理结果会与源模型不一致。

### 2.4 关键注意事项

| 问题 | 说明 |
| ---- | ---- |
| ReflectionPad1d方向 | 源模型使用`ReflectionPad1d((0, 64))`，仅在右侧填充64个样本。若误用`ReflectionPad1d(32)`（左右各32），STFT输出会错误 |
| LSTM状态处理 | 解码器对LSTM输出`h`应用Dropout→ReLU→Conv→Sigmoid得到语音概率，但传给下一帧的隐藏状态必须是原始的`h`（未经Dropout/ReLU），否则会累积误差 |
| 固定时间维度 | 编码器输出shape为[1, 128, 1]（时间维度固定为1），使用reshape替代squeeze以避免生成Shape/Gather算子 |
| 输出维度 | 解码器输出使用reshape(-1, 1)替代squeeze(1).mean(1, keepdim=True)，因为T'=1是固定的，两者数学等价 |