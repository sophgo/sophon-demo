# 模型量化

更多模型量化教程请参考《TPU-MLIR 开发参考手册》的「模型量化」（请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的 SDK 中获取）。

## 1. 注意事项
### 1.1 量化数据集
本例程使用 `datasets/cali` 下的 NYU Depth V2 train 集标定图片（抽样 200 张）作为量化数据集，量化数据集应尽量涵盖测试场景和类别，量化时可尝试不同的 `--input_num` 进行量化以获得最优的量化精度。

### 1.2 前处理对齐
量化数据集的预处理应该和推理测试的预处理保持一致（RGB、stretch resize 到 768×768、/255、float32 输入 768×768）。`gen_int8bmodel_mlir.sh` 中 `model_transform.py` 已通过 `--mean 0 --scale 0.0039216 --pixel_format rgb` 与推理端预处理对齐（**不使用** `--keep_aspect_ratio`，对应官方 val 的 `LetterBox(scale_fill=True)` 无 padding 拉伸），`run_calibration.py` 直接读取原图即可。

## 2. 量化流程

本例程的 INT8 量化已封装进 `gen_int8bmodel_mlir.sh`，核心步骤为：

```bash
# 1. 模型转换（得到 yolo26s_depth_1b.mlir）
model_transform.py \
    --model_name yolo26s_depth \
    --model_def ../models/onnx/yolo26s-depth.onnx \
    --input_shapes [[1,3,768,768]] \
    --mean 0.0,0.0,0.0 \
    --scale 0.0039216,0.0039216,0.0039216 \
    --pixel_format rgb \
    --mlir yolo26s_depth_1b.mlir

# 2. 生成 calibration table
run_calibration.py yolo26s_depth_1b.mlir \
    --dataset ../datasets/cali \
    --input_num 200 \
    --part_asymmetric \
    --cali_method percentile9999 \
    --fp_type F32 \
    -o yolo26s_depth_cali_table

# 3. 部署为 INT8 BModel
model_deploy.py \
    --mlir yolo26s_depth_1b.mlir \
    --quantize INT8 \
    --chip bm1688 \
    --calibration_table yolo26s_depth_cali_table \
    --model yolo26s_depth_int8_1b.bmodel
```

> **说明**：本例程模型侧直接输出绝对尺度度量深度图（log 深度还原 + log-affine 标定 + 4× bilinear 上采样已全部烘焙进图），无需手工拆分敏感层/维护 qtable，直接执行 `gen_int8bmodel_mlir.sh` 即可完成量化。若量化精度不达预期，可调整 `--input_num`（增大标定样本数）或改用 `Middle`/`hidden optimized` 等校准策略后重试。