# SAM3 模型导出与编译

本文件说明如何从 SAM3 官方 PyTorch 权重 `sam3.pt` 导出 ONNX，再编译为可在 SOPHON TPU 上运行的 BModel。**多数用户无需走全流程**——`scripts/download.sh` 已提供预编译好的 FP16 BModel 与导出好的 ONNX，直接用即可；仅当需要重新导出/编译或更换精度时参考本文件。

## 1. 准备工作

### 1.1 权重
- `sam3.pt`（~3.45 GB）来自 Meta 官方仓库 [facebookresearch/sam3](https://github.com/facebookresearch/sam3)，权重在 HuggingFace (`facebook/sam3`) gated 托管，需申请下载：
  ```bash
  pip install huggingface_hub
  huggingface-cli login            # 用已获批的 HF token
  huggingface-cli download facebook/sam3 sam3.pt --local-dir ./models
  ```
- 放到 `models/sam3.pt`。**仅重新导出 ONNX 时需要**；运行预编译 bmodel 推理不需要。

### 1.2 环境
| 环境 | 用途 | 依赖 |
|------|------|------|
| **Host**（Python 3.10） | 导出 ONNX（grounding / text / 1008 trunk+neck） | `sam3` 包 + torch + onnx + onnxruntime≥1.20 |
| **tpu-mlir Docker**（`sophgo/tpuc_dev:v3.4`） | 导出 504 ViT/Neck ONNX + 编译所有 BModel | tpu-mlir（`model_transform.py` / `model_deploy.py`） |

> Docker 内 `/workspace` 即宿主 `/home/lihengfang/work/git_commits`。504 的 ViT/Neck 导出脚本硬编码了 `/workspace/...` 路径，须在 Docker 内执行；grounding/text 导出脚本用宿主绝对路径，在 host 执行。

### 1.3 分辨率约定
- **504×504 → SoC**（SE7-32/BM1684X SoC、SE9/BM1688）：NPU 峰值 ~1.6 GB，交付集。
- **1008×1008 → PCIe**：NPU 需 3.48 GB，超 SE7-32 SoC 的 3 GB 限制，仅 PCIe。
- Grounding + Text Encoder 在交付流水线中：504 为 SoC 交付集；1008 为 PCIe f16 增项（`--grounding` 开启，SoC 显存不够）。

## 2. 导出 ONNX

### 2.1 504×504（SoC，交付集）

**ViT 5-part** → `models/onnx_504/sam3_vit_part{0..4}.onnx`（+ `.onnx.data` 外置权重）

交付的 5-part ONNX 由以下方式之一生成（均需 sam3.pt）：
```bash
# 方式 A：先导 trunk 再图分割（Docker 内）
python tools/export_onnx.py --start_export --resolution 504  # → onnx_504/sam3_vit_trunk.onnx
python tools/split_vit_onnx.py --execute                   # trunk → 5 part（注：默认 1008 token，504 需调 grid）

# 方式 B：分段导出（Docker 内，--blocks_per_part 8 即 5-part 布局）
python tools/export_vit_fine.py --blocks_per_part 8 --start_export   # 注：shape 默认 1008，504 需改
```
> 交付集 `onnx_504.zip` 已含正确 5-part，多数用户直接 `download.sh` 拉取。自行从 scratch 重导需注意把 token 数从 5184(72²) 改为 1296(36²)。

**Neck FPN** → `models/onnx_504/sam3_neck_combined.onnx`（Docker 内）
```bash
python tools/export_neck_onnx.py --start_export --resolution 504  # 输入 (1,1024,36,36), opset 14
```

**Grounding Encoder + Decoder** → `models/onnx_grounding_504/`（host）
```bash
python tools/export_grounding_all.py --grid 36 --output_dir ../models/onnx_grounding_504
# → sam3_grounding_encoder.onnx (opset 17) + sam3_grounding_decoder.onnx (opset 17)
```

**Text Encoder**（可选）→ `models/onnx/sam3_text_encoder.onnx`（host）
```bash
python tools/export_text_encoder_onnx.py                   # 输入 (1,32) int64 token IDs, opset 16
```
> Text Encoder 的 bmodel 已预编译随交付集提供（`BM1684X_504/grounding/sam3_text_encoder_f16_1b.bmodel`），运行推理无需再导出。

### 2.2 1008×1008（PCIe）

```bash
python tools/export_onnx.py --start_export --resolution 1008 --output_dir ../models/onnx  # → onnx/sam3_vit_trunk.onnx
python tools/export_neck_onnx.py --start_export --resolution 1008 --output_dir ../models/onnx  # → onnx/sam3_neck_combined.onnx (72×72)
```

**Grounding Encoder + Decoder**（PCIe f16 增项，可选）→ `models/onnx_grounding_1008/`（host）
```bash
python tools/export_grounding_all.py --grid 72 --output_dir ../models/onnx_grounding_1008
# → sam3_grounding_encoder.onnx + sam3_grounding_decoder.onnx（grid=72, N=5184）
```
Text Encoder 与 504 共用同一个 ONNX（`models/onnx/sam3_text_encoder_static.onnx`，分辨率无关）。

## 3. 编译 BModel

进入 tpu-mlir Docker，在例程目录执行：

### 3.1 504×504 SoC（交付集，FP16）
```bash
# BM1684X (SE7 等)
./scripts/gen_bmodel.sh --res 504 --chip bm1684x --mode f16
# → models/BM1684X_504/{vit,neck,grounding}/*_f16_1b.bmodel（9 个 bmodel）

# BM1688 (SE9 等，单核，图同构)
./scripts/gen_bmodel.sh --res 504 --chip bm1688 --mode f16
# → models/BM1688_504/...
```

### 3.2 1008×1008 PCIe
```bash
./scripts/gen_bmodel.sh --res 1008 --chip bm1684x --mode f32   # 或 --mode f16
# → models/BM1684X/{vit,neck}/*_{f32|f16}_1b.bmodel（仅骨干）

# 全流程（含 Grounding/Text，PCIe f16 增项）
./scripts/gen_bmodel.sh --res 1008 --chip bm1684x --mode f16 --grounding
# → 额外生成 models/BM1684X/grounding/{sam3_grounding_encoder,sam3_grounding_decoder,sam3_text_encoder}_f16_1b.bmodel
```

`--grounding` 默认关闭（0=仅骨干）。开启后编译 Grounding Encoder（grid=72, N=5184）、Decoder、Text Encoder 三个 f16 bmodel。SoC 显存不够（Encoder O(N²) 16× + ViT 3GB），仅 PCIe 可用。推理需 `--streaming` 标志避免 OOM（见 README §4）。

### 3.3 1008×1008 INT8（SoC 落地）

F16 1008 单个 ViT Part NPU 显存 3.45GB，超 SE7-32 SoC 3GB 限制。INT8 量化把 Part 1-4 压到 1.35GB/Part，配合流式加载（part1-4 逐个 load→run→free）峰值显存 = 1.35GB，可落 SoC。

**前置：校准集**。`tools/prepare_int8_cali_1008.py` 用 onnxruntime 逐 Part 跑真实激活，生成各 Part 的校准 npz（key 严格等于模型输入名，`run_calibration.py` 强校验）：

```bash
# host 上（onnxruntime），输入 N 张真实图 + 1008 onnx 目录
python3 tools/prepare_int8_cali_1008.py \
    --image_dir datasets --onnx_dir models/onnx \
    --cali_dir datasets/cali_data_1008 --num 24 --resolution 1008
# → datasets/cali_data_1008/{part0,part1,part2,part3,part4,neck}/<i>.npz
```

**编译**（tpu-mlir Docker 内）：

```bash
./scripts/gen_bmodel.sh --res 1008 --chip bm1684x --mode int8
# → models/BM1684X/vit/sam3_vit_part{1..4}_int8_1b.bmodel（Part 0/Neck 无 int8，回退 f16）
```

`gen_bmodel.sh` 的 int8 路径已内置：`run_calibration.py --tune_num 0`（跳过 tune，24 样本 KLD 已够；tune 对 0-dim 张量崩 `cali_math.py:265`）+ `model_deploy.py --opt 1`（Part 3/4 用默认 opt=2 时 address-assign 阶段 `LmemAllocator.cpp:1645 llvm_unreachable`，opt=1 绕过）。校准集规模可通过 `CALI_INPUT_NUM` / `CALI_TUNE_NUM` 环境变量覆盖。

**提精度（可选）**：剩余精度杠杆为 `--tune_num > 0` 主动调优逐 op 量化阈值。需先修 tpu-mlir `python/calibration/cali_math.py:265` 的 0-dim 崩（2 行 size guard，Python-only 无需 rebuild），再 `CALI_TUNE_NUM=8 ./scripts/gen_bmodel.sh --res 1008 --chip bm1684x --mode int8`，耗时 ~1-2hr。改 tpu-mlir 源码前先报用户。

**推理**（host 验证 / SoC 部署）：

```bash
# PCIe 全常驻（显存充裕，更快）
python3 python/sam3_vit_infer.py --model_dir models/BM1684X --precision int8 --resolution 1008
# SoC 流式（part1-4 load→run→free，峰值 1.35GB 落 3GB）
python3 python/sam3_vit_infer.py --model_dir models/BM1684X --precision int8 --resolution 1008 --streaming
```

**完整流水线入口（`sam3_infer.py --precision int8`）**：自动接线 1008 + `models/BM1684X`，跑 ViT+Neck 骨干，跳过 grounding（INT8 1008 仅骨干，不含 Grounding/Text），输出 FPN 特征 npz（backbone-only，不依赖 sam3 源码）：

```bash
python3 python/sam3_infer.py --precision int8 --image datasets/truck.jpg
# → results/sam3_backbone_int8.npz（fpn_feat_{0,1,2} + fpn_pos_{0,1,2}）
```

**SoC 全骨干流式（`sam3_infer.py --streaming`）**：`--streaming` 让 `SAM3Engines` 不常驻 ViT part1-4，改由 `SAM3ViTEncoder` 逐 part load→run→free（part0/neck 小，仍常驻）。峰值显存 = max 单 part = 1.35GB，落 SoC 3GB。PCIe 上流式比全常驻慢约 900ms/帧（part1-4 reload 开销），SoC 上是落地的唯一路径：

```bash
# SoC（SE9/BM1688 等）全骨干流式
python3 python/sam3_infer.py --precision int8 --streaming --image datasets/truck.jpg
```

> **SE9-16（BM1688 SoC）实测**：`--precision int8 --streaming` exit 0，FPN 输出 (1,256,288/144/72)² 三级特征正确，ViT 10504ms + Neck 7230ms/帧，仅 part0 常驻，不 OOM（SoC 3.4G RAM 下推理+npz 落盘均通过）。BM1688 int8 bmodel 经 `DOWNLOAD_1008_INT8_BM1688=1 ./scripts/download.sh` 拉取。

> `gen_bmodel.sh --res 504` 依次编译 ViT 5 part + Neck + Grounding enc/dec + Text enc，消耗 `onnx_504/` 与 `onnx_grounding_504/`；`--res 1008` 编译 ViT 4 part + Neck（消耗 `onnx/`），加 `--grounding` 额外编译 Grounding enc/dec + Text enc（消耗 `onnx_grounding_1008/`，PCIe f16 增项）。

## 4. 算子兼容性与已知限制

| 问题 | 影响 | 现状 |
|------|------|------|
| **presence_logit_dec 丢弃**（TPU-MLIR `Save:424` 阻塞） | 仅影响绝对置信度 ~0.12，不影响排序/框/mask | 规避：导出时该输出不接，推理端不依赖 |
| **BF16 / F32 bmodel 不可用** | 同 `Save:424`（`si32→f32` Cast 触发 `TPU_DYNAMIC→A53` hang） | 仅交付 FP16 |
| **Grounding Decoder presence-feature** | tpu-mlir v3.4 对该输入处理有限 | 导出时把 presence feature 塞进 `reference_boxes` 通道（见 `export_grounding_onnx.py` 内 workaround） |
| **ONNX IR v10 + 外置 data** | 旧 onnxruntime 读不了 | 需 onnxruntime ≥ 1.20；`.onnx` 与 `.onnx.data` 须同目录 |
| **ViT trunk 1.7 GB** | `model_transform.py` 加载整图峰值 ~25 GB | 拆 5 part 分别编译（见 README §4 拆分说明） |

## 5. 一致性 / 性能工具（可选）

- `tools/consistency_harness.py`：TPU vs PyTorch 源分阶段 cos + 端到端 IoU。
- `tools/sam3_source_ref.py`：源模型推理 ground truth。
- `tools/grounding_model_profile.py`：Grounding 子模型 profiling。

## 6. 参考
- [SAM3 官方仓库](https://github.com/facebookresearch/sam3)
- [TPU-MLIR 快速入门手册](https://developer.sophgo.com/site/index.html?categoryActive=material)
- [TPU-MLIR 环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)
