# Python例程

## 目录

- [Python例程](#python例程)
  - [目录](#目录)
  - [1. 环境准备](#1-环境准备)
    - [1.1 SoC平台](#11-soc平台)
  - [2. 推理测试](#2-推理测试)
    - [2.1 参数说明](#21-参数说明)
    - [2.2 测试图片](#22-测试图片)
    - [2.3 ONNX模式（纯CPU推理）](#23-onnx模式纯cpu推理)
  - [3. 精度测试](#3-精度测试)
  - [4. 流程图](#4-流程图)

python目录下提供了Python例程，具体情况如下：

| 序号 | Python例程        | 说明                                              |
| ---- | ---------------- | ------------------------------------------------- |
| 1    | sam3_infer.py    | SAM3 完整推理流水线（ViT+Neck+Grounding+后处理）     |
| 2    | sam3_vit_infer.py | ViT 视觉编码器 TPU 推理（单独模块测试）              |
| 3    | sam3_neck_infer.py | Neck FPN TPU 推理（单独模块测试）                  |

## 1. 环境准备

### 1.1 SoC平台

算能的SoC平台（如SE、SM系列边缘设备）在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包。但除此之外您还需要交叉编译安装sophon-sail，具体可参考[交叉编译安装sophon-sail](../../../docs/Environment_Install_Guide.md#42-交叉编译安装sophon-sail)。

在运行之前您需要先安装一些Python依赖，可以通过pip3直接安装：

```bash
pip3 install -r requirements.txt
```

## 2. 推理测试

### 2.1 参数说明

`sam3_infer.py` 的参数说明如下：

```bash
usage: sam3_infer.py [-h] [--model_dir MODEL_DIR] [--precision {f32,f16}]
                     [--dev_id DEV_ID] [--resolution RESOLUTION] [--grid GRID]
                     [--image IMAGE] [--prompt PROMPT]
                     [--score_thresh SCORE_THRESH] [--output OUTPUT]
                     [--ckpt_path CKPT_PATH] [--bpe_path BPE_PATH]
                     [--mode {bmodel,onnx}]

SAM3 Full Inference Pipeline

optional arguments:
  -h, --help            show this help message and exit
  --model_dir MODEL_DIR Path to compiled bmodel directory (default: models/BM1684X_504)
  --precision {f32,f16} Model precision (default: f16)
  --dev_id DEV_ID       TPU device ID (default: 0)
  --resolution RESOLUTION Input resolution (default: 504)
  --grid GRID           Feature grid size, 36 for 504, 72 for 1008 (default: 36)
  --image IMAGE         Input image path
  --prompt PROMPT       Text prompt for detection, e.g. "a truck", "a dog", "groceries"
  --score_thresh SCORE_THRESH Score threshold for detection (default: 0.3)
  --output OUTPUT       Output image path (default: results/sam3_detection.jpg)
  --ckpt_path CKPT_PATH SAM3 PyTorch checkpoint path
  --bpe_path BPE_PATH   BPE tokenizer vocabulary path
  --mode {bmodel,onnx}  bmodel=TPU sail inference, onnx=onnxruntime CPU inference
```

### 2.2 测试图片

以 `datasets/truck.jpg` 为例，使用文本提示 "a truck" 进行目标检测：

```bash
cd sample/SAM3
python3 python/sam3_infer.py \
  --image datasets/truck.jpg \
  --prompt "a truck" \
  --model_dir models/BM1684X_504 \
  --precision f16 \
  --mode bmodel \
  --output results/sam3_truck_bmodel.jpg
```

输出结果示例如下（框内为检测到的目标，文字为提示词）：

<div style="text-align: center;">
  <img src="../results/sam3_truck_bmodel.jpg" alt="truck detection" style="width: 65%;">
</div>

其他测试图片示例：

```bash
# 狗
python3 python/sam3_infer.py --image datasets/dog.jpg --prompt "a dog" --output results/sam3_dog_bmodel.jpg
# 杂货架
python3 python/sam3_infer.py --image datasets/groceries.jpg --prompt "groceries" --output results/sam3_groceries_bmodel.jpg
```

<div style="text-align: center;">
  <img src="../results/sam3_dog_bmodel.jpg" alt="dog detection" style="width: 65%;">
  <img src="../results/sam3_groceries_bmodel.jpg" alt="groceries detection" style="width: 65%;">
</div>

### 2.3 ONNX模式（纯CPU推理）

当 TPU 不可用时，可使用 `--mode onnx` 进行纯 CPU 推理（不需要 TPU 设备）：

```bash
python3 python/sam3_infer.py \
  --image datasets/truck.jpg \
  --prompt "a truck" \
  --mode onnx \
  --output results/sam3_truck_onnx.jpg
```

> **注意**：ONNX 模式首次运行需要加载 PyTorch 模型（~3.3GB），耗时约 40s。后续可加缓存优化。

## 3. 精度测试

精度对比使用 `tools/consistency_harness.py`，同时运行 ONNX/bmodel 和 PyTorch 源码推理，逐阶段比较一致性：

```bash
cd sample/SAM3
PYTHONPATH=~/work/git_commits/developer/sam3 python3 tools/consistency_harness.py \
  --mode bmodel \
  --image datasets/truck.jpg \
  --prompt "a truck" \
  --shared_input \
  2>&1 | tee results/consistency_truck_bmodel.log
```

参数说明：
- `--mode {bmodel,onnx}`：选择推理后端
- `--shared_input`：向两边馈入相同输入张量，消除预处理差异
- 输出包含：逐阶段 shape/stats/cos、per-layer gnd_dec hs cos (L0-L5)、端到端 top-1 IoU/score_diff/mask_IoU

完整精度报告见 `results/` 目录下的 consistency 日志和 `../README.md` 的精度测试章节。

## 4. 流程图

SAM3 完整推理流水线如下：

<div style="text-align: center;">
  <img src="../pics/sam3_pipeline.png" alt="SAM3 pipeline" style="width: 80%;">
</div>

```
[Image] → Preprocess → ViT Encoder (5 parts) → Neck FPN
                                                       ↓
[Text]  → Text Encoder ──────────────────────→ Grounding Encoder
                                                       ↓
                                              Grounding Decoder (6 layers)
                                                       ↓
                                              CPU Post-process (scoring + box refine)
                                                       ↓
                                              [Boxes + Scores + Masks]
```
