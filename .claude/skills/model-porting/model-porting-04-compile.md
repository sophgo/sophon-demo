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
