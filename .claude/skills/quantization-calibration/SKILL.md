---
name: quantization-calibration
description: 量化校准全流程。在 tpu-mlir Docker 内把 ONNX/FP32 模型编译成 INT8 bmodel，覆盖校准集准备、qtable 混精度调优、精度验证（逐层 cos/npz 对比）、OOM 规避。说「按照 quantization-calibration skill 量化 XXX 模型」即可。
---

# 量化校准

> 目的：把 FP32/FP16 模型量化为 INT8 bmodel，在可接受精度损失下获得 TPU 加速收益。覆盖校准集准备、量化编译、精度验证、精度回退排查的完整闭环。

## 适用范围

- TPU-MLIR 工具链（`model_transform` → `run_calibration` → `model_deploy`）
- 目标芯片：BM1684X / BM1688 / CV186X
- 量化模式：INT8 对称/非对称、W8F16（权重 INT8 激活 F16，免校准）

## 前置条件

| 检查 | 命令 | 期望 |
|------|------|------|
| tpu-mlir Docker | `docker ps -a --filter name=tpu_mlir` | 存在 `sophgo/tpuc_dev:v3.4` 容器 |
| 环境初始化 | `source /workspace/git_commits/tpu-mlir/envsetup.sh` | `which model_transform.py` 能找到 |
| ONNX 模型 | `ls *.onnx` | 原始 ONNX 已导出且可用 |
| 校准数据 | `ls datasets/cali/` | 校准图片/数据已准备 |

## 流程（6 阶段）

### 阶段 1：校准集准备

1. 从训练集随机抽取 100~500 张代表性样本。校准集应涵盖测试场景的各类别、各类光照/尺度。
2. **预处理必须与推理一致**：量化校准数据的预处理（resize/归一化/颜色空间）必须和最终推理用的一致，否则量化参数失配，精度损失放大。
3. 制作校准集：
   - 图像模型：用 `gen_npz.py` 或手动脚本把预处理后的图片打包成 `.npz`（每张一个 key，value = 预处理后的 NCHW float32 数组）。
   - 多输入模型：每个输入一个 npz 或合并到一个 npz 的多个 key。
   - 音频模型：把特征（如 FBANK/Mel）打包成 npz/npy。
4. 校准集大小权衡：数量多→校准更稳但耗时增加；通常 100~500 张。OOM 时减小 batch 或减数量。

### 阶段 2：model_transform（ONNX → MLIR）

```bash
model_transform.py \
  --model_name yolov8s \
  --model_def yolov8s.onnx \
  --input_shapes "[[1,3,640,640]]" \
  --mean 0.0,0.0,0.0 \
  --scale 0.0039216,0.0039216,0.0039216 \
  --keep_aspect_ratio \
  --pixel_format rgb \
  --chip bm1684x \
  --test_input cali.npz \
  --test_result yolov8s_top_outputs.npz \
  --tolerance 0.99,0.99
```

**关键参数：**
- `--mean` / `--scale`：必须和推理预处理一致。图像归一化 `/255` 对应 `--scale 0.0039216,0.0039216,0.0039216 --mean 0,0,0`。
- `--input_shapes`：多输入用嵌套列表 `[[$1,3,640,640],[$1,2]]`。
- `--keep_aspect_ratio`：letterbox 预处理时加。
- `--tolerance`：FP32 参考对比的余弦相似度阈值，建议 0.99。

**常见坑：**
- `dynamic_axes` 导致 `TPU_DYNAMIC` subnet → 后续量化/推理行为不一致。尽量用 `--input_shapes` 固定 batch，避免动态。
- 多输入模型的第二个输入（如 orig_size）是 int32，不需要 mean/scale，但 model_transform 仍需在 input_shapes 中声明。

### 阶段 3：run_calibration（量化校准）

```bash
run_calibration.py yolov8s.mlir \
  --dataset cali.npz \
  --num_group 3 \
  --tune_num 0 \
  --chip bm1684x \
  -o yolov8s_cali_table
```

**关键参数：**
- `--num_group`：分组量化粒度，3 为常用值。
- `--tune_num`：调优轮数，0 = 快速校准。精度不够时增加到 50~100（慢但可能提升）。
- `--chip`：`bm1684x` / `bm1688` / `cv186x`。

**OOM 规避：**
| 场景 | 解决 |
|------|------|
| 大模型（ViT/Large）单样本校准 OOM | 减小 `--num_group`；用 W8F16 代替 INT8（免校准）|
| 1008 分辨率模型 OOM | 用 `--tune_num 0`；确保 swap ≥ 64GB；减小校准集数量 |
| `swap` 不足 | `docker run` 时加 `--memory-swap` 或 host 加 swap 文件 |

**W8F16 替代方案：** 当 INT8 校准反复 OOM 或精度严重不达标时，W8F16（权重 INT8 + 激活 F16）免校准，精度损失通常 < 2%，是稳妥的降级方案。`gen_bmodel.sh --mode w8f16` 通常已支持。

### 阶段 4：model_deploy（编译 bmodel）

```bash
# INT8 对称量化
model_deploy.py \
  --model_name yolov8s \
  --mlir yolov8s.mlir \
  --calibration_table yolov8s_cali_table \
  --chip bm1684x \
  --quantize INT8 \
  --quantize_table yolov8s_qtable \
  --test_input cali.npz \
  --test_reference yolov8s_top_outputs.npz \
  --tolerance 0.85,0.55 \
  --model yolov8s_int8_1b.bmodel

# W8F16（免校准）
model_deploy.py \
  --model_name yolov8s \
  --mlir yolov8s.mlir \
  --chip bm1684x \
  --quantize W8F16 \
  --model yolov8s_w8f16_1b.bmodel
```

**精度验证阈值：**
- `--tolerance 0.85,0.55`：INT8 量化后逐层余弦相似度 ≥ 0.85、相对误差 ≤ 0.55 是经验可接受范围。
- 模型内部解码（如 DETR 类）对量化更敏感，可能需要 0.9,0.3。
- 验证不通过 → model_deploy 报 `npz compare FAILED` → 进入阶段 5 排查。

**关键标志：**
- `--quantize INT8`：标准 INT8 量化。
- `--quantize W8F16`：权重 INT8 激活 F16，不需 calibration_table。
- `--quantize_table`：混精度 qtable，指定哪些层保持 F16。
- `--chip bm1684x`：部分 MLIR 版本不支持 bm1688/cv186x 做 qtable，可用 bm1684x 代替生成的 qtable（通用）。

### 阶段 5：精度回退排查

当 INT8 精度不达标时，按以下顺序排查：

**步骤 1：定位回退层（逐层 cos）**

```bash
# model_deploy 阶段加 --debug，或直接 npz_tool 逐层对比
npz_tool.py compare \
  yolov8s_bm1684x_int8_sym_tpu_outputs.npz \
  yolov8s_top_outputs.npz \
  --tolerance 0.8,0.5 --except - -vv
```

查看每层 cos 值，找出 cos 骤降的层（通常 < 0.9 即异常）。

**步骤 2：qtable 混精度（敏感层保持 F16）**

1. 用 `mlir2onnx.py -m xxx.mlir -o xxx.onnx` 转出 ONNX，用 Netron 查看网络结构。
2. 用 `fp_forward.py` 生成 qtable：
   ```bash
   fp_forward.py yolov8s.mlir \
     --fpfwd_outputs /model.22/dfl/conv/Conv_output_0_Conv \
     --chip bm1684x \
     -o yolov8s_qtable
   ```
3. 手动编辑 qtable，把敏感层改为 F16：
   ```
   # qtable 格式：层名 精度类型
   /model.22/dfl/conv/Conv_output_0_Conv F16
   /model.22/Sigmoid_output_0_Sigmoid F16
   ```
4. 重新 model_deploy（带 `--quantize_table yolov8s_qtable`），验证精度是否提升。

**YOLO 系列经验值：**
| 模型 | 敏感层 | 建议策略 |
|------|--------|----------|
| YOLOv5 | 最后 3 个 conv 层 | `--fpfwd_outputs` 指定最后 3 个 conv |
| YOLOv8 | DFL conv + Sigmoid | 坐标回归 conv + Sigmoid 设为 F16 |
| DETR 类（D-FINE 等） | decoder + 后处理 | 通常建议 W8F16 而非 INT8 |

**步骤 3：th_method 调优**

如果 qtable 无效，尝试不同的门限策略：
```bash
run_calibration.py xxx.mlir \
  --dataset cali.npz \
  --th_method ADMM \  # 可选: ADMM, MAX, PERCENTILE9999
  --chip bm1684x
```
推荐顺序：ADMM → MAX → PERCENTILE9999。

**步骤 4：非对称量化（INT8 asym）**

如果对称量化精度差，尝试非对称：
```bash
model_deploy.py ... --quantize INT8 --asym  # 注意：不是所有模型都支持 asym
```
注意：部分模型 asym 会崩（精度更差），需实测对比。

### 阶段 6：端到端验证

1. 在 SoC/PCIe 上用 INT8 bmodel 跑推理，对比 FP16 基线：
   - 检测模型：mAP 或逐图框对齐（参考 precision-alignment skill）
   - 分类模型：Top-1 accuracy
   - 分割模型：mIoU
   - 语音模型：CER/WER
2. 精度达标 → 交付 INT8 bmodel。不达标 → 回到阶段 5 或改用 W8F16。
3. 记录量化结论：INT8 cos/mAP、W8F16 cos/mAP、FP16 基线，形成对比表。

## 常见问题速查

| 现象 | 原因 | 解决 |
|------|------|------|
| `npz compare FAILED` | 量化后逐层误差超阈值 | 查 cos 骤降层，加 qtable 混精度 |
| 校准 OOM | 模型太大 / 校准集太多 | W8F16 免校准；减 `--tune_num 0`；加 swap |
| `TPU_DYNAMIC` subnet | ONNX 有 dynamic_axes | 固定 batch 重导出 ONNX |
| INT8 比对称量化差 | asym 不适用 | 改回 sym；用 ADMM th_method |
| qtable 层名对不上 | ONNX 版本不同导致层名变 | 重新 `model_transform` + `fp_forward` 生成 |
| `--quant_input` 不生效 | 混淆 `--quant_input` 和 `--quant_input_int8` | 输入 INT8 化用 `--quant_input`；`--quant_input_int8` 是 no-op |
| model_deploy 报 `context_dir` 碰撞 | `--model` 没带 `.bmodel` 后缀 | 加后缀 `--model xxx.bmodel` |

## 执行纪律

- 量化前确认 ONNX 的 dynamic_axes 已去除（固定 batch），否则后续链路可能发散。
- 校准集预处理必须和推理一致——这是量化精度的头号因素。
- 精度不达标时按「逐层 cos 定位 → qtable 混精度 → th_method 调优 → W8F16 降级」的顺序尝试，不要跳步。
- 清理：量化完成后删除中间产物（.mlir/.npz/.profile/.json/per-model 目录），保留 bmodel + cali_table + qtable。
- 汇报用中文，结论先行。
