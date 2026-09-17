# 模型量化

更多模型量化教程请参考《TPU-MLIR 开发参考手册》的「模型量化」（请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的 SDK 中获取）。

## 1. 注意事项
### 1.1 量化数据集
本例程使用 `datasets/cali` 下的 Cityscapes 标定图片（抽样 12 张，6 个城市各 2 张）作为量化数据集，量化数据集应尽量涵盖测试场景和类别，量化时可尝试不同的 iterations 进行量化以获得最优的量化精度。

### 1.2 前处理对齐
量化数据集的预处理应该和推理测试的预处理保持一致（RGB、letterbox pad 114、/255、float32 输入 1024×2048）。`gen_int8bmodel_mlir.sh` 中 `model_transform.py` 已通过 `--mean 0 --scale 0.0039216 --keep_aspect_ratio --pixel_format rgb` 与推理端预处理对齐，`run_calibration.py` 直接读取原图即可。

## 2. 量化流程

本例程的 INT8 量化已封装进 `gen_int8bmodel_mlir.sh`，核心步骤为：

```bash
# 1. 模型转换（得到 yolo26s_1b.mlir）
model_transform.py \
    --model_name yolo26s \
    --model_def ../models/onnx/yolo26s-sem.onnx \
    --input_shapes [[1,3,1024,2048]] \
    --mean 0.0,0.0,0.0 \
    --scale 0.0039216,0.0039216,0.0039216 \
    --keep_aspect_ratio \
    --pixel_format rgb \
    --mlir yolo26s_1b.mlir

# 2. 生成 calibration table
run_calibration.py yolo26s_1b.mlir \
    --dataset ../datasets/cali \
    --input_num 12 \
    --part_asymmetric \
    --cali_method percentile9999 \
    --fp_type F32 \
    -o yolo26s_cali_table

# 3. 部署为 INT8 BModel
model_deploy.py \
    --mlir yolo26s_1b.mlir \
    --quantize INT8 \
    --chip bm1688 \
    --calibration_table yolo26s_cali_table \
    --model yolo26s_int8_1b.bmodel
```

> **说明**：本例程模型侧直接输出逐像素类别图（bilinear 上采样 + argmax 已烘焙进图），argmax 为单调算子、量化时无需手工拆分敏感层/维护 qtable，直接执行 `gen_int8bmodel_mlir.sh` 即可完成量化。若量化精度不达预期，可调整 `--input_num`（增大标定样本数）或改用 `Middle`/`hidden optimized` 等校准策略后重试。