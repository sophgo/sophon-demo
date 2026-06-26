---
name: model-porting-04-compile
description: 编译 ONNX → BModel。多芯片多精度批量编译，处理编译错误并回溯修复。
---

# 步骤 04：编译 BModel

## 前置

步骤 03 已完成（ONNX 文件在 `models/onnx/`）。

## 提示词

```
将 ONNX 编译为 BModel，生成 scripts/gen_bmodel_mlir.sh：

1. model_transform.py（ONNX → MLIR）：
   model_transform.py --model_name [name] --model_def [onnx] \
     --input_shapes [[b,c,h,w]] --input_types float32 --mlir [name].mlir

2. run_calibration.py（仅 INT8/INT8_4b）：
   run_calibration.py [name].mlir --dataset [cali_data] --input_num 200 \
     -o [name]_cali_table

3. model_deploy.py（MLIR → BModel）：
   model_deploy.py --mlir [name].mlir --quantize [F32/F16/INT8/W4BF16] \
     --chip [bm1684x/bm1688/cv186x] --model [name]_[precision]_[batch]b.bmodel

4. 编译目标：模板中指定的所有 芯片 × 精度 × batch 组合
5. 每个 BModel 用 bmrt_test --bmodel [path] 验证可加载
6. 编译失败时分析日志中的 "unsupported op"，回溯步骤 03 修复 ONNX

输出编译脚本 scripts/gen_bmodel_mlir.sh（或按精度拆分为多个脚本）。
```

## 预期输出

- `models/[chip]/[name]_[precision]_[batch]b.bmodel`
- `scripts/gen_fp32bmodel_mlir.sh`、`gen_fp16bmodel_mlir.sh`、`gen_int8bmodel_mlir.sh`

## 内联知识：编译参数速查

| 参数 | 说明 | 值 |
|------|------|-----|
| `--quantize` | 量化类型 | F32 / F16 / BF16 / INT8 / W4BF16（即 INT8_4b） |
| `--chip` | 目标芯片 | bm1684x / bm1688 / cv186x |
| `--num_core` | TPU 核数 | 1 |
| `--dynamic` | 动态 shape | 支持可变 batch |
| `--customization_format` | INT8 精度格式 | BM1684X 用 `bm1684x` |

### 输出命名规范

`[model_name]_[precision]_[batch]b.bmodel`

例：`mobilenetv4_fp16_4b.bmodel`、`mobilenetv4_int8_1b.bmodel`

## Debug

| 问题 | 排查方向 |
|------|---------|
| "unsupported op" | 查看具体算子名，回步骤 03 替换或降 opset |
| 编译成功但 bmrt_test 加载失败 | libsophon 版本与 TPU-MLIR 版本不匹配 |
| INT8 校准精度差 | 确保校准数据集与模型输入预处理一致 |
| 编译 OOM | 减小 max_input_len 或 batch size |
| bmodel 输出与 onnx 不一致 | 用 onnxruntime 和 sail 在**同一输入**下对比输出；若 onnx 对而 bmodel 错，多为算子编译退化（如自定义 MHA 替换 SDPA），回步骤 03 用真实算子（torch 1.13）重新导出 |
| 检测 AP≈0 / 全某一类且分数极低 | **不要归因于 `--mean/--scale`**（见下），首选排查文本嵌入/预处理是否正确 |

## 内联知识：`--mean`/`--scale` 的真实作用（重要）

`model_transform.py` 的 `--mean`/`--scale` **只是写入 mlir 文件的元信息，仅在校准量化（INT8）时使用**，**不会烤入 bmodel 作为运行时计算 op**。因此：
- FP32/FP16 bmodel 推理时**不会对输入做 mean/scale**，输入值域完全由 onnx 模型本身决定（看 onnx 首层）。
- 推理侧 python 的归一化（如 `/255`）是**唯一**的输入归一化，不存在"`--scale` + python `/255` 双重归一化"。
- 若 onnx 期望 [0,1] 输入（首层 Conv 直连、无 1/255 常量），python 喂 `/255` 即可，`--mean/--scale` 加不加都不影响 FP32/FP16 结果。
- AP=0 的真正原因通常是：文本嵌入子模型产出错位（开放词汇模型）、预处理值域/通道顺序错、或某算子被 TPU-MLIR 误编译——按步骤 06 的隔离法排查，**不要怀疑 `--mean/--scale`**。

## 实战备忘

- **多子模型**：主模型 + 文本编码器等子模型分别 model_transform/model_deploy，bmodel 文件名建议带版本/子模型标识（如 `yoloworld_v2_fp16_1b.bmodel`、`clip_text_vitb32_bm1684x_f16_1b.bmodel`）避免与其它版本混用。
- **编译中间产物**：model_deploy 会在 scripts/ 下生成大量 `.mlir/.npz/.json/.profile/.prototxt`，提交 git 前清理（只保留 `.sh`）。
