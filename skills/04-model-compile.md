# Skill 4: 模型编译 (ONNX → BModel)

## 目标
使用 TPU-MLIR 将 ONNX 模型编译为 BModel，使其能在 BM1684X TPU 上运行。

## 执行步骤

### 4.1 准备 TPU-MLIR 环境
```bash
# 进入 TPU-MLIR Docker 环境
cd /path/to/tpu-mlir
source envsetup.sh
```

### 4.2 模型转换 (ONNX → MLIR)
```bash
# FP32 模型
model_transform.py \
    --model_name model_sub1 \
    --model_def 子模型1.onnx \
    --input_shapes [[1,100,560],[1]] \
    --input_types float32,int32 \
    --mlir model_sub1.mlir
```

### 4.3 生成校准表 (仅 INT8 需要)
```bash
# 准备校准数据集
# 运行校准
run_calibration.py model_sub1.mlir \
    --dataset cali_data \
    --input_num 100 \
    -o model_sub1_cali_table
```

### 4.4 模型部署 (MLIR → BModel)
```bash
# FP32 编译
model_deploy.py \
    --mlir model_sub1.mlir \
    --quantize F32 \
    --chip bm1684x \
    --model model_sub1_fp32_10b.bmodel \
    --num_core 1 \
    --dynamic

# INT8 编译
model_deploy.py \
    --mlir model_sub1.mlir \
    --quantize INT8 \
    --calibration_table model_sub1_cali_table \
    --chip bm1684x \
    --model model_sub1_int8_10b.bmodel \
    --num_core 1 \
    --dynamic
```

### 4.5 批量编译脚本
```bash
# 子模型1
model_transform.py ... --mlir 子模型1.mlir
model_deploy.py ... --model submodel1_fp32.bmodel

# 子模型2
model_transform.py ... --mlir 子模型2.mlir
model_deploy.py ... --model submodel2_fp32.bmodel

# 子模型3
model_transform.py ... --mlir 子模型3.mlir
model_deploy.py ... --model submodel3_fp32.bmodel
```

## 关键参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--chip` | 目标芯片 | bm1684x |
| `--quantize` | 量化类型 | F32 / INT8 |
| `--dynamic` | 支持动态 shape | true (需要动态 batch/T) |
| `--num_core` | TPU 核心数 | 1 |
| `--model` | 输出文件名 | `{name}_fp32_{batch}b.bmodel` |
| `--max_input_len` | 最大输入长度 | 根据数据集确定 |

## 检查清单

- [ ] 所有 ONNX 模型编译成功
- [ ] BModel 文件大小合理（FP32 应接近 ONNX 大小）
- [ ] 使用 bmrt_test 验证 BModel 可加载
- [ ] 对每个 BModel 确认输入输出 shape
- [ ] dynamic bmodel 测试不同输入长度

## 常见问题

1. **编译失败 "unsupported op"**: 检查 ONNX 中是否有 TPU-MLIR 不支持的算子
2. **动态 shape 不生效**: 检查 `--dynamic` 标志和 ONNX 的 dynamic_axes
3. **INT8 精度差**: 检查校准数据集是否具有代表性
4. **模型过大超内存**: 减小 max_input_len 或 batch size
