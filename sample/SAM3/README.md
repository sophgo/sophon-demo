# SAM3 (Segment Anything Model 3)

## 目录

- [SAM3 (Segment Anything Model 3)](#sam3-segment-anything-model-3)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
    - [2.1 目录结构说明](#21-目录结构说明)
    - [2.2 SDK特性](#22-sdk特性)
  - [3. 准备模型与数据](#3-准备模型与数据)
  - [4. 模型编译](#4-模型编译)
    - [模型拆分说明](#模型拆分说明)
  - [5. 例程测试](#5-例程测试)
    - [Python例程](#python例程)
  - [6. 精度测试](#6-精度测试)
    - [6.1 测试方法](#61-测试方法)
    - [6.2 测试结果](#62-测试结果)
  - [7. 性能测试](#7-性能测试)
    - [7.1 bmrt\_test](#71-bmrt_test)
    - [7.2 程序运行性能](#72-程序运行性能)
  - [8. FAQ](#8-faq)
  - [附录](#附录)
    - [SoC 部署](#soc-部署)
    - [移植进度](#移植进度)
    - [已知限制](#已知限制)
    - [参考](#参考)

## 1. 简介

​SAM3（Segment Anything Model 3 with Concepts）是Meta提出的统一基础模型，支持基于文本、点、框和掩码提示的图像和视频分割。与前代SAM2相比，SAM3新增了开放词汇文本提示（可处理270K+概念）和DETR风格的目标检测能力。本例程对[​SAM3官方开源仓库](https://github.com/facebookresearch/sam3)的模型和算法进行移植，使之能在SOPHON BM1684X/BM1688上进行推理测试。

**模型参数**:
- 总参数量: ~840M
- 输入图像尺寸: 1008×1008 (PCIe), 504×504 (SoC)
- 预处理: Resize(504,504 或 1008,1008) → Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])

> **SoC适配说明**: SoC 的 TPU 内存在默认分区下划分为若干 ION heap，bmodel 推理通常只能使用 npu heap，故「SoC 3GB 限制」实指默认 npu heap 分区，并非物理 TPU 内存。SE7-32 (BM1684X SoC) TPU 8.3GB = npu 2.36GB + vpu 2.87GB + vpp 3.00GB；SE9-16 (BM1688 SoC, 16G DDR) 仅 npu + vpp 两 heap（无 vpu），默认 npu 2GB。1008×1008 F16 单 ViT Part 需 3.45GB 超过默认 npu heap；因此针对 SoC 默认分区采用 504×504 低分辨率模型（NPU 峰值约 1.6GB）。1008 落 SoC 的路径因芯片而异：
> - **SE7-32 (BM1684X)**：INT8 单 Part 压到 1.35GB，流式加载（part1-4 逐个 load→run→free）峰值 2.10GB 可 fit 默认 npu heap；用 `memory_edit.sh -c -npu 6144 -vpu 1024 -vpp 1024` 重分配后还可全常驻 INT8（峰值 5.96GB，比流式快 20–32%）。
> - **SE9-16 (BM1688)**：INT8 单 Part NPU 运行显存约 **3.6GB**（远大于 BM1684X 的 1.35GB，neuron 激活 3.23GB），默认 2GB npu heap 直接 OOM，须先用 `memory_edit.sh -c -npu 6144 -vpu 0 -vpp 2048 bm1688_se9v1_16G` 扩到 6GB 才能跑流式（峰值 ≤4.04GB）；**全常驻不可行**——5 Part+Neck 需 ≥13.7GB，扩到 12GB 仍 OOM（npu peak 10.51GB 触顶），16G DDR 亦容不下，故 BM1688 上流式是 INT8 1008 落地的唯一路径。详见 §7.1。

## 2. 特性

### 2.1 目录结构说明

```
./SAM3
├── docs
│   └── export_bmodel.md                              # 本例程中bmodel编译中的算子问题说明文档
├── pics
├── python                                             # 存放Python例程及其README
│   ├── sam3_infer.py                                   # SAM3 完整推理流水线（ViT+Neck+Grounding+后处理）
│   ├── sam3_vit_infer.py                               # ViT 编码器 TPU 推理
│   ├── sam3_neck_infer.py                              # Neck FPN TPU 推理
│   ├── simple_tokenizer.py                             # CLIP tokenizer（独立实现，免装 sam3 包）
│   ├── requirements.txt                                # Python 依赖
│   └── README.md                                       # SAM3 Python例程的说明文件
├── README.md                                          # 本例程的中文指南
├── scripts
│   ├── download.sh                                    # 模型和数据集下载脚本
│   └── gen_bmodel.sh                                  # BModel 一键编译脚本（504/1008，BM1684X/BM1688，Docker内）
└── tools
    ├── export_onnx.py                                 # ViT trunk → ONNX（--resolution 504/1008，Docker内）
    ├── export_vit_fine.py                             # ViT 分段导出（5 part，Docker内）
    ├── split_vit_onnx.py                              # ONNX 图分割（trunk → 5 part）
    ├── export_neck_onnx.py                            # Neck FPN → ONNX（--resolution 504/1008，Docker内）
    ├── export_grounding_all.py                        # Grounding Encoder+Decoder → ONNX（一键）
    ├── export_grounding_onnx.py                       # Grounding Encoder/Decoder → ONNX（分别）
    ├── export_text_encoder_onnx.py                    # TextEncoder → ONNX
    └── ...                                            # 一致性/性能工具见 docs/export_bmodel.md
```

### 2.2 SDK特性

* 支持BM1684X (x86 PCIe、SoC)、BM1688 (SoC)
* ViT视觉编码器部分支持FP16、FP32模型编译和推理
* Neck FPN部分支持FP16、FP32模型编译和推理
* Grounding Encoder / Decoder 部分支持FP16、FP32模型编译和推理
* Text Encoder部分支持FP16、FP32模型编译和推理
* 支持基于OpenCV和SAIL的Python推理
* 支持文本提示的目标检测（text-to-box）
* 支持几何提示的推理（点、框）
* 支持图片批量推理

**注意：
本repo中完整推理流程需要多个bmodel串联运行：ViT编码器（5部分）→ Neck FPN → Grounding Encoder → Grounding Decoder → Text Encoder → 后处理（CPU numpy）。各引擎在初始化时全部加载，推理时直接复用，消除每次推理的加载开销。**

## 3. 准备模型与数据

​本例程在`scripts`目录下提供了相关模型和数据的下载脚本`download.sh`，您也可以自己准备模型和数据集，并参考[4. 模型编译](#4-模型编译)进行模型转换。

```bash
# 安装unzip，若已安装请跳过，非ubuntu系统视情况使用yum或其他方式安装
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```

下载的模型包括：
```
./models
├── BM1684X_504                                    # BM1684X(SE7等) 504×504 FP16 bmodel
│   ├── vit
│   │   ├── sam3_vit_part0_f16_1b.bmodel              # ViT Part 0 fp16 bmodel (504×504)
│   │   ├── sam3_vit_part1_f16_1b.bmodel              # ViT Part 1 fp16 bmodel (504×504)
│   │   ├── sam3_vit_part2_f16_1b.bmodel              # ViT Part 2 fp16 bmodel (504×504)
│   │   ├── sam3_vit_part3_f16_1b.bmodel              # ViT Part 3 fp16 bmodel (504×504)
│   │   └── sam3_vit_part4_f16_1b.bmodel              # ViT Part 4 fp16 bmodel (504×504)
│   ├── neck
│   │   └── sam3_neck_f16_1b.bmodel                   # Neck FPN fp16 bmodel (504×504)
│   └── grounding
│       ├── sam3_grounding_encoder_f16_1b.bmodel       # Grounding Encoder fp16 bmodel
│       ├── sam3_grounding_decoder_f16_1b.bmodel       # Grounding Decoder fp16 bmodel
│       └── sam3_text_encoder_f16_1b.bmodel            # Text Encoder fp16 bmodel
├── BM1688_504                                    # BM1688(SE9等) 504×504 FP16 bmodel（单核，结构同 BM1684X_504）
│   ├── vit
│   │   ├── sam3_vit_part0_f16_1b.bmodel
│   │   ├── sam3_vit_part1_f16_1b.bmodel
│   │   ├── sam3_vit_part2_f16_1b.bmodel
│   │   ├── sam3_vit_part3_f16_1b.bmodel
│   │   └── sam3_vit_part4_f16_1b.bmodel
│   ├── neck
│   │   └── sam3_neck_f16_1b.bmodel
│   └── grounding
│       ├── sam3_grounding_encoder_f16_1b.bmodel
│       ├── sam3_grounding_decoder_f16_1b.bmodel
│       └── sam3_text_encoder_f16_1b.bmodel
├── onnx_504                                       # 504×504 ViT 5 part + Neck ONNX（用于自行编译 bmodel）
│   ├── sam3_vit_part0.onnx  (+ sam3_vit_part0.onnx.data)
│   ├── sam3_vit_part1.onnx  (+ sam3_vit_part1.onnx.data)
│   ├── sam3_vit_part2.onnx  (+ sam3_vit_part2.onnx.data)
│   ├── sam3_vit_part3.onnx  (+ sam3_vit_part3.onnx.data)
│   ├── sam3_vit_part4.onnx  (+ sam3_vit_part4.onnx.data)
│   └── sam3_neck_combined.onnx  (+ sam3_neck_combined.onnx.data)
├── onnx_grounding_504                            # Grounding ONNX（用于自行编译）
│   ├── sam3_grounding_encoder.onnx               # Grounding Encoder ONNX
│   └── sam3_grounding_decoder.onnx               # Grounding Decoder ONNX
├── post_process_weights.npz                      # 后处理权重（dot_prod_scoring + bbox_embed，bmodel 推理必需）
└── seg_head_weights.npz                          # mask decoder head 权重（bmodel 推理必需）
```

> 注：bmodel 推理只需 `BM1684X_504/`（或 `BM1688_504/`）下的 bmodel + 上述两个 `.npz` 后处理权重，**不需要 `sam3.pt`**。两个 `.npz` 由 `download.sh` 随交付集下发（从 `sam3.pt` 一次性提取的 head 权重）；仅在缺失时推理代码才会回退到从 `sam3.pt` 提取。`sam3.pt` 原始 PyTorch 权重仅"自行重新导出 onnx"时需要（HuggingFace 申请下载，见第 4 节）。
> 注：`onnx_504` 下每个 `.onnx` 的权重存于同名 `.onnx.data` 外置文件，两者需同目录存放；Text Encoder 的 onnx 源在 `models/onnx/`（由 `tools` 从 `sam3.pt` 导出），其 bmodel 已预编译在 `BM1684X_504/grounding/`、`BM1688_504/grounding/` 下，运行推理无需再准备该 onnx。

下载的数据包括：
```
./datasets
├── truck.jpg                                         # 测试图片1
├── groceries.jpg                                     # 测试图片2
└── dog.jpg                                           # 测试图片3
```

## 4. 模型编译

导出的模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。建议使用TPU-MLIR编译BModel。

模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的"3. 编译ONNX模型"(请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的SDK中获取)。

直接使用官方的torch模型文件进行导出和编译，会出现算子不兼容的问题，修改方法可参考[bmodel导出](docs/export_bmodel.md)。

本例程在`scripts`目录下提供了一键编译脚本`gen_bmodel.sh`，通过`--res`指定分辨率、`--chip`指定目标平台、`--mode`指定量化方式，即可编译出该配置下完整推理流水线所需的全部 BModel：

- 生成 FP16 BModel（504×504 SoC 交付集，BM1684X）

```bash
./scripts/gen_bmodel.sh --res 504 --chip bm1684x --mode f16
```

​执行上述命令会在`models/BM1684X_504/vit`下生成 5 个 ViT Part bmodel，在`models/BM1684X_504/neck`下生成 Neck FPN bmodel，在`models/BM1684X_504/grounding`下生成 Grounding Encoder、Grounding Decoder 和 Text Encoder bmodel，即 504×504 分辨率完整推理流水线的 FP16 BModel。

- 生成 FP16 BModel（504×504 SoC，BM1688 单核）

```bash
./scripts/gen_bmodel.sh --res 504 --chip bm1688 --mode f16
```

​执行上述命令会在`models/BM1688_504/{vit,neck,grounding}`下生成与 BM1684X_504 同构的 FP16 BModel（单核），即 BM1688 平台的 504×504 完整推理流水线 BModel。

- 生成 FP32/FP16 BModel（1008×1008 PCIe）

```bash
./scripts/gen_bmodel.sh --res 1008 --chip bm1684x --mode f32   # 或 --mode f16
```

​执行上述命令会在`models/BM1684X/vit`下生成 ViT Part bmodel，在`models/BM1684X/neck`下生成 Neck FPN bmodel（默认仅骨干 ViT+Neck）。

- 生成 FP16 BModel（1008×1008 PCIe 全流程，含 Grounding/Text）

```bash
./scripts/gen_bmodel.sh --res 1008 --chip bm1684x --mode f16 --grounding
```

​加 `--grounding` 后额外在 `models/BM1684X/grounding/` 下编译 Grounding Encoder、Grounding Decoder、Text Encoder 三个 f16 bmodel，即 1008×1008 全流程（ViT+Neck+Grounding+Text）的 FP16 BModel。此为 **PCIe f16 增项**——SoC 显存不够（Grounding Encoder O(N²) 随 grid 16× 增长 + ViT 顶 3GB），仅 PCIe 可用。

- 生成 INT8 BModel（1008×1008，SoC 落地）

```bash
# 需先准备校准集（见 docs/export_bmodel.md §3.3）
./scripts/gen_bmodel.sh --res 1008 --chip bm1684x --mode int8
```

​INT8 把 1008 单个 ViT Part 的 NPU 显存从 F16 的 3.45GB 压到 1.35GB（weight 114MB + runtime 1.23GB），使 1008 骨干能在 SE7-32 SoC（默认 npu heap 2.36GB）上以流式加载运行（part0 常驻 + part1-4 逐个 load→run→free，实测峰值 2.10GB）。全常驻（SAM2 风格 part0-4 + Neck 同时加载）int8 需 5.96GB 超默认 npu heap；但 SE7-32 TPU 总内存 8.3GB（npu+vpu+vpp 三 heap），可用 `memory_edit.sh` 把 npu heap 调到 6GB（见 §7.1），全常驻即可落 SoC 且比流式快 20–32%（省去 eMMC 重载）。**默认分区下流式是落 SoC 的路径；改内存后全常驻亦可。** Part 0 与 Neck 无 INT8 变体，自动回退 F16（Part 0 含预处理、Neck 为小 FPN，量化收益小风险大）；INT8 Part 1-4 bmodel 各 114–135MB（vs F16 227MB）。精度：24 样本校准下 Neck 输出 cos ≈ 0.89–0.90 vs F16（不保真，见 §6.3）。对比 W8F16（仅压权重、激活仍 F16）精度 0.91 同档但显存 ~3.3GB 不解决默认 SoC npu heap 限制，故选 INT8。

**推理**：`sam3_infer.py --precision int8` 自动接线 1008 + `models/BM1684X`，跑 ViT+Neck 骨干，跳过 grounding（INT8 1008 仅骨干，不含 Grounding/Text），输出 FPN 特征 npz（backbone-only，不依赖 sam3 源码）；`sam3_vit_infer.py --precision int8 --resolution 1008 --streaming` 为 SoC 流式单模块入口。

```bash
python3 python/sam3_infer.py --precision int8 --image datasets/truck.jpg   # → results/sam3_backbone_int8.npz
```

- 1008×1008 PCIe 全流程推理（f16，含 grounding 出框）

1008 全流程需 `--streaming` 标志：ViT part1-4 逐个 load→run→free（part0 常驻），降低峰值显存。不加 `--streaming` 时所有引擎同时加载（5 ViT + Neck + Text Enc + Gnd Enc + Gnd Dec），峰值显存超设备上限，part4 加载时 OOM（`bm_alloc_gmem failed, size=3.48GB`）。

```bash
# 1008 全流程（ViT+Neck+Grounding+Text），需 --streaming 避免 OOM
python3 python/sam3_infer.py --image datasets/truck.jpg --prompt "a truck" \
    --model_dir models/BM1684X --precision f16 --resolution 1008 --mode bmodel --streaming
# → results/sam3_1008_truck_full.jpg（图像 + 检测框）
```

测试结果（PCIe x86 + BM1684X F16，2026-07-22）：

| 图片 | prompt | score | 框 (cx, cy, w, h) | 端到端耗时 |
|------|--------|-------|-------------------|-----------|
| truck | "a truck" | 0.5874 | (0.509, 0.459, 0.875, 0.491) | ~7.4s |
| dog | "a dog" | 0.5284 | (0.252, 0.737, 0.525, 0.556) | ~7.7s |
| groceries | "groceries" | 0.4861 | (0.631, 0.458, 0.340, 0.375) | ~7.5s |

框均落在目标上，score 合理。ViT 骨干 ~5.9s 占主导（1008 分辨率 4× token），grounding ~0.4s。结果图见 `results/sam3_1008_{truck,dog,groceries}_full.jpg`。

### 模型拆分说明

SAM3 ViT主干网络包含32个Transformer Block，完整ONNX模型约1.7GB。由于`model_transform.py`在转换时需要加载整个计算图到内存（峰值约25GB），超出可用内存限制，因此将模型拆分为5个部分分别编译：

**1008×1008 (PCIe)**

| 部分 | 内容 | 输入 Shape | 输出 Shape |
|------|------|-----------|------------|
| Part 0 | patch_embed + pos_embed + ln_pre | [1, 3, 1008, 1008] | [1, 5184, 1024] |
| Part 1 | ViT blocks 0-7 | [1, 5184, 1024] | [1, 5184, 1024] |
| Part 2 | ViT blocks 8-15 | [1, 5184, 1024] | [1, 5184, 1024] |
| Part 3 | ViT blocks 16-23 | [1, 5184, 1024] | [1, 5184, 1024] |
| Part 4 | ViT blocks 24-31 | [1, 5184, 1024] | [1, 5184, 1024] |

**504×504 (SoC SE7-32)**

| 部分 | 内容 | 输入 Shape | 输出 Shape |
|------|------|-----------|------------|
| Part 0 | patch_embed + pos_embed + ln_pre | [1, 3, 504, 504] | [1, 1296, 1024] |
| Part 1 | ViT blocks 0-7 | [1, 1296, 1024] | [1, 1296, 1024] |
| Part 2 | ViT blocks 8-15 | [1, 1296, 1024] | [1, 1296, 1024] |
| Part 3 | ViT blocks 16-23 | [1, 1296, 1024] | [1, 1296, 1024] |
| Part 4 | ViT blocks 24-31 | [1, 1296, 1024] | [1, 1296, 1024] |

推理时按顺序串联执行，5个Part的输出依次作为下一个Part的输入。

## 5. 例程测试

### [Python例程](./python/README.md)

## 6. 精度测试

### 6.1 测试方法

精度对比使用 ，同时运行 ONNX/bmodel 和 PyTorch 源码推理，逐阶段比较张量一致性：

usage: consistency_harness.py [-h] [--image IMAGE] [--prompt PROMPT]
                              [--model_dir MODEL_DIR]
                              [--precision {f16,f32,bf16,int8,f16hp}]
                              [--mode {bmodel,onnx}] [--resolution RESOLUTION]
                              [--ckpt_path CKPT_PATH] [--bpe_path BPE_PATH]
                              [--abs_thresh ABS_THRESH]
                              [--rel_thresh REL_THRESH] [--shared_input]
consistency_harness.py: error: unrecognized arguments: onnx datasets/dog.jpg datasets/groceries.jpg

参数说明：
- ：选择推理后端（bmodel=TPU sail, onnx=onnxruntime CPU）
- ：向两边馈入相同输入张量，消除预处理差异
- 输出包含：逐阶段 shape/stats/cos、per-layer gnd_dec hs cos (L0-L5)、端到端 top-1 IoU/score_diff/mask_IoU

### 6.2 测试结果

**测试环境**: BM1684X F16 bmodel + ONNX FP32 CPU 双模式，3 张测试图 (truck/dog/groceries)，以 PyTorch 源码推理为参考基准。

**端到端结果**：

| 图片 | Mode | gnd_dec hs cos | Top-1 | Box IoU | Mask IoU |
|------|------|---------------|-------|---------|----------|
| truck | bmodel F16 | 0.9814 | 144→144 ✅ | 0.9978 | 0.9906 |
| truck | onnx FP32 | 0.9816 | 144→144 ✅ | — | — |
| dog | bmodel F16 | 0.9519 | 104→104 ✅ | 0.9986 | 0.9965 |
| dog | onnx FP32 | 0.9532 | 104→104 ✅ | — | — |
| groceries | bmodel F16 | 0.9192 | 121→90 ❌ | 0.0 | 0.0 |
| groceries | onnx FP32 | 0.9203 | 121→90 ❌ | — | — |

> **说明**：ONNX 模式的 Box IoU/Mask IoU 与 bmodel 模式基本一致（bmodel F16 忠实复现 ONNX FP32 输出，cos 差异 ≤0.002），故不再重复列出。

**Grounding Decoder 逐层 hs cos 诊断**：

| Layer | Truck | Dog | Groceries |
|-------|-------|-----|-----------|
| L0 | 0.9975 | 0.9885 | 0.9850 |
| L1 | 0.9888 | 0.9640 | 0.9471 |
| L2 | 0.9892 | 0.9574 | 0.9247 |
| L3 | 0.9744 | 0.9405 | 0.9029 |
| L4 | 0.9733 | 0.9388 | 0.8895 |
| L5 | 0.9663 | 0.9280 | 0.8679 |

发散模式为逐层累积（每层 ~0.01-0.02 cos），无单层跳崖。truck/dog 的 top query 分数 margin 大，累积误差不足以翻转 top-1；groceries 的 queries 分数 near-tied，微小扰动即可翻转。

完整 per-layer 数据和 per-stage diff 报告见 。

### 6.3 INT8 1008 精度（ViT+Neck only）

1008×1008 仅含 ViT+Neck（无 Grounding/Text），INT8 量化针对 ViT Part 1-4（Part 0/Neck 回退 F16）。24 样本校准（`--input_num 24 --tune_num 0`）下，以 F16 1008 全链为参考基准：

| 对比项 | cos vs F16 |
|--------|-----------|
| Part 1 隔离 | 0.770 |
| Part 2 隔离 | 0.628 |
| Part 3 隔离 | 0.629 |
| Part 4 隔离 | 0.641 |
| ViT 末态（Part 0-4 链式累积） | 0.35 |
| **Neck FPN 最终输出** | **0.89–0.90** |

逐 Part 隔离 cos 偏低且链式累积退化严重（INT8 激活量化经 32 个 Transformer Block 残差链滚雪球，ViT 末态落到 0.35），但 Neck FPN 把方向拉回 0.89–0.90。1008 交付物即 Neck 输出，0.89 与 W8F16（0.91）同档，**不保真**（504 F16 基线 0.99+）。W8F16 仅压权重（559M vs 925M）、激活仍 F16，1008 显存 ~3.3GB 仍超默认 SoC npu heap（2.36GB），不解决落地问题；INT8 压激活才是落 SoC 的路。若需提精度，剩余杠杆为 `--tune_num > 0` 主动调优逐 op 量化阈值（需先修 tpu-mlir `cali_math.py` 0-dim 崩，见 docs/export_bmodel.md §3.3）。

## 7. 性能测试

### 7.1 bmrt_test

使用bmrt_test测试模型的理论性能：

```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684X_504/vit/sam3_vit_part0_f16_1b.bmodel --devid 0
bmrt_test --bmodel models/BM1684X_504/vit/sam3_vit_part1_f16_1b.bmodel --devid 0
bmrt_test --bmodel models/BM1684X_504/grounding/sam3_grounding_encoder_f16_1b.bmodel --devid 0
```

测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。

测试各个模型的理论推理时间，结果如下（FP16；SE7=BM1684X_504 测试日期 2026-07-09，SE9=BM1688_504 测试日期 2026-07-10）：

|    测试平台   | 测试模型                                                     | calculate time(ms) |
| -----------   | ------------------------------------------------------------ |  ----------------- |
|   SE7-32      | BM1684X_504/vit/sam3_vit_part0_f16_1b.bmodel                 |          5.8       |
|   SE7-32      | BM1684X_504/vit/sam3_vit_part1_f16_1b.bmodel                 |        272.8       |
|   SE7-32      | BM1684X_504/vit/sam3_vit_part2_f16_1b.bmodel                 |        272.7       |
|   SE7-32      | BM1684X_504/vit/sam3_vit_part3_f16_1b.bmodel                 |        272.8       |
|   SE7-32      | BM1684X_504/vit/sam3_vit_part4_f16_1b.bmodel                 |        273.0       |
|   SE7-32      | BM1684X_504/neck/sam3_neck_f16_1b.bmodel                     |         30.6       |
|   SE7-32      | BM1684X_504/grounding/sam3_grounding_encoder_f16_1b.bmodel   |         28.6       |
|   SE7-32      | BM1684X_504/grounding/sam3_grounding_decoder_f16_1b.bmodel   |         20.4       |
|   SE7-32      | BM1684X_504/grounding/sam3_text_encoder_f16_1b.bmodel        |         13.7       |
|   SE9-16      | BM1688_504/vit/sam3_vit_part0_f16_1b.bmodel                  |          9.9       |
|   SE9-16      | BM1688_504/vit/sam3_vit_part1_f16_1b.bmodel                  |        664.7       |
|   SE9-16      | BM1688_504/vit/sam3_vit_part2_f16_1b.bmodel                  |        664.7       |
|   SE9-16      | BM1688_504/vit/sam3_vit_part3_f16_1b.bmodel                  |        664.7       |
|   SE9-16      | BM1688_504/vit/sam3_vit_part4_f16_1b.bmodel                  |        664.7       |
|   SE9-16      | BM1688_504/neck/sam3_neck_f16_1b.bmodel                      |         87.4       |
|   SE9-16      | BM1688_504/grounding/sam3_grounding_encoder_f16_1b.bmodel    |         58.1       |
|   SE9-16      | BM1688_504/grounding/sam3_grounding_decoder_f16_1b.bmodel    |         42.0       |
|   SE9-16      | BM1688_504/grounding/sam3_text_encoder_f16_1b.bmodel        |         50.3       |

> **测试说明**：  
> 1. 性能测试结果具有一定的波动性；
> 2. 上述数据通过`sail.Engine.process`多次取平均测得（SE7 5次、SE9 20次），与bmrt_test结果基本一致；
> 3. ViT部分总计: SE7 = 5.8 + 272.8×4 ≈ 1097ms；SE9 = 9.9 + 664.7×4 ≈ 2669ms（5个part串联推理）；
> 4. Grounding encoder/decoder为多输入模型，其数据来自端到端运行时统计；
> 5. SE9 (BM1688) 数据为单核(num_core=1)编译测试；BM1688 支持双核(num_core=2)，ViT 等大模型双核吞吐约可翻倍（本表未测）。BM1688 单核 F16 算力低于 BM1684X，故 ViT 单 part SE9≈664.7ms vs SE7≈272.8ms。

**1008×1008 INT8（PCIe x86 + BM1684X，测试日期 2026-07-22）**：

| 测试模型 | calculate time(ms) |
|----------| ----------------- |
| vit/sam3_vit_part0_f16_1b.bmodel（Part 0 回退 F16） | 87 |
| vit/sam3_vit_part1_int8_1b.bmodel | 656 |
| vit/sam3_vit_part2_int8_1b.bmodel | 655 |
| vit/sam3_vit_part3_int8_1b.bmodel | 653 |
| vit/sam3_vit_part4_int8_1b.bmodel | 656 |

端到端 ViT 串联（`sam3_vit_infer.py --precision int8 --resolution 1008`）：全常驻 2736ms/帧、流式（`--streaming`）3054ms/帧（~12% 重加载开销，换默认 SoC npu heap fit）；流式与全常驻输出 bitwise 一致（maxdiff=0, cos=1.0），仅慢约 900ms/帧（part1-4 reload 开销）。完整流水线入口 `sam3_infer.py --precision int8`（ViT+Neck 骨干，backbone-only）：~4060ms/帧（ViT 2894 + Neck 1144，含预处理），输出 FPN 特征 npz。

**1008×1008 INT8（SE9-16 BM1688 SoC，流式 `--streaming`，npu heap 6GB，测试日期 2026-08-24）**：

| 测试模型 | calculate time(ms) |
|----------| ----------------- |
| vit/sam3_vit_part0_f16_1b.bmodel（Part 0 回退 F16） | 9.4 |
| vit/sam3_vit_part1_int8_1b.bmodel | ~1750 |
| vit/sam3_vit_part2_int8_1b.bmodel | ~1750 |
| vit/sam3_vit_part3_int8_1b.bmodel | ~1750 |
| vit/sam3_vit_part4_int8_1b.bmodel | ~1750 |
| neck/sam3_neck_f16_1b.bmodel（Neck 回退 F16） | 242 |

SE9 流式全骨干 e2e（`sam3_infer.py --precision int8 --streaming`，npu heap 已扩至 6GB）：ViT ≈10.4s + Neck ≈5.6s ≈ 16.1s/帧（0.062 FPS），npu peak ≤4.04GB（part0 F16 常驻 + 1 个 INT8 Part 的 neuron+coeff），FPN 三级特征 (1,256,288/144/72)² 正确。表内 per-part ~1750ms 为纯 TPU 计算时间，ViT e2e 10.4s 与 4×1750≈7.0s 的差值约 3.4s 为流式逐个重载 part1-4 bmodel（各 ~114MB，eMMC I/O）的开销；Neck e2e 5.6s 远大于纯计算 242ms，主因是 FPN 三级特征的 CPU 后处理（top-k/精修 numpy）。

> **SE9-16 (BM1688) 显存注记（与 SE7-32 / BM1684X 不同）**：BM1688 上单个 INT8 ViT Part 的 NPU 运行显存约 **3.6GB**（weight 114MB + neuron 激活 3.23GB），远大于 BM1684X 的 1.35GB——BM1688 INT8 激活 neuron 仍是全精度张量、且单核路径 neuron_mem 占比更高。故 **默认 npu heap（2GB）直接 OOM**，必须先用 `memory_edit.sh -c -npu 6144 -vpu 0 -vpp 2048 bm1688_se9v1_16G` 把 npu heap 扩到 6GB（写 `/boot/boot.itb` 后重启生效）才能跑流式。**全常驻不可行**：5 Part 全加载 + Neck 需 ≥13.7GB，即便把 npu heap 扩到 12GB 仍 OOM（part4 推理时 `bm_alloc_gmem failed, size=3.23GB`，npu peak 10.51GB 触顶），16G DDR 亦无法容纳 npu 14GB + vpp + 系统，故 BM1688 上**流式是 INT8 1008 落地的唯一路径**（对比 SE7-32 全常驻 6GB heap 可行，见下表）。

**1008×1008 INT8（SE7-32 BM1684X SoC，测试日期 2026-08-21）**：

SE7-32 默认 TPU 分区：npu 2.36GB + vpu 2.87GB + vpp 3.00GB（合计 8.23GB）。INT8 1008 落 SE7-32 两种模式均验证通过（FPN 三级特征 (1,256,288/144/72)² 正确，三图 FPN norm 与 PCIe 全常驻一致）：

| 模式 | 命令 | npu heap 配置 | 峰值显存 | truck / dog / groceries e2e (ViT+Neck) |
|------|------|--------------|---------|----------------------------------------|
| 流式（默认分区，不改内存） | `sam3_infer.py --precision int8 --streaming` | 默认 2.36GB | 2.10GB（part0 F16 常驻 + 1 INT8 Part） | 9159 / 9365 / 7972 ms |
| 全常驻（改内存后） | `sam3_infer.py --precision int8`（不加 `--streaming`） | 6.0GB（memory_edit） | 5.96GB（part0-4 + Neck 全常驻） | 6414 / 6382 / 6372 ms |

> SE7-32 全常驻比流式快 20–32%（PCIe 仅快 ~12%）：SoC 流式每次从 eMMC 重载 part1-4 四个 bmodel（各 ~130MB），I/O 开销远大于 PCIe；全常驻省去重载但需 npu heap ≥ 5.96GB，默认 2.36GB 不够，须先用 `memory_edit.sh -c -npu 6144 -vpu 1024 -vpp 1024` 重分配（npu 可访问 vpu/vpp 内存，改完写 `/boot/emmcboot.itb` 后重启生效，参考 `../Qwen/README.md`）。流式与全常驻输出 bitwise 一致（三图 FPN norm 逐位相同）。

### 7.2 程序运行性能

参考[Python例程](python/README.md)运行程序，并查看统计的预处理时间、推理时间、后处理时间。目前SAM3仅支持1 batch的fp16模型。前处理部分包含对图像的resize和normalize操作，后处理部分包含点积评分、MLP框精修和mask解码（numpy实现，无需CPU PyTorch模型）。

测试端到端推理性能结果如下（时间单位为ms），测试结果有一定波动性：

|  测试平台   |      测试程序       |                          bmodel                           | preprocess_time |  vit_time  | neck_time | text_enc_time | gnd_enc_time | gnd_dec_time | postprocess_time | mask_time |
|----------|-------------------|-----------------------------------------------------------|----------------|-----------|----------|--------------|-------------|-------------|-----------------|-----------|
|   SE7-32    |  sam3_infer.py  | BM1684X_504 FP16 (ViT 5part+Neck+TextEnc+GndEnc+GndDec) |       3       |   1116    |   951    |    9555      |     30      |     20      |       19        |   13394   |
|   SE9-16    |  sam3_infer.py  | BM1688_504 FP16 (ViT 5part+Neck+TextEnc+GndEnc+GndDec) |    41   |   2695   |   2156   |    6342     |     61      |     43      |       46        |   N/A*   |

> **测试说明**：  
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. 图片分辨率对解码时间影响较大（504×504），推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异；
> 4. 采用SAM2风格的引擎常驻模式（load-once），ViT 5个引擎 + Neck + 3个Grounding引擎全部在初始化时加载，推理时直接复用，消除每次推理的bmodel加载开销；
> 5. text_enc_time 含首次导入sam3模块链的开销（~9.5s，ARM SoC磁盘I/O），持久化进程下可消除；TPU纯推理仅13.7ms；
> 6. postprocess_time=19ms（点积评分+MLP框精修），mask_time=13394ms（numpy mask解码器CPU后处理，pixel_decoder 6层上采样 + mask_predictor）；
> 7. 后处理权重首次从checkpoint提取约需122s并缓存为seg_head_weights.npz，后续运行直接加载（<1ms）；
> 8. SE9 (BM1688) mask_time 标 N/A*：SE9 用户态 RAM 仅约 851MB（BM1688 SoC 内存大量划给 TPU），全分辨率(1200×1800) numpy mask 解码器（~2.2GB 数组）内存不足无法跑通，测试时用 `SAM3_SKIP_MASK=1` 跳过；mask 解码为纯 numpy CPU 代码，与芯片无关，可参考 SE7 同代码的 13394ms。检测仍正常出框（truck score 0.53, cx=0.499/cy=0.465/w=0.850/h=0.484，结果图见 results/sam3_detection.jpg）；
> 9. SE9 各 TPU 阶段相对 SE7 普遍 2.1–2.4×（BM1688 单核 F16 算力低于 BM1684X，与 7.1 一致）；唯 text_enc_time 反而 6342 < SE7 9555，是因 SE9 用独立 simple_tokenizer（绕开 sam3 包导入链）冷启动更轻，非芯片优势——SE9 TPU 纯文本推理 50.3ms 仍慢于 SE7 13.7ms（见 7.1）。SE9 preprocess=41ms（CPU 图像解码+resize，SE7 仅 3ms）。

## 8. FAQ

问题请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。

## 附录

### SoC 部署

BM1684X SoC (SE7-32) 与 PCIe 使用相同的 BModel 文件，无需重新编译。将编译好的 `models/BM1684X_504/`（或 `BM1688_504/`）连同 `python/`、`datasets/` 拷到 SoC 即可：

```bash
# 在 host 上把交付集传到 SoC（示例 IP/账号按实际改）
scp -r models/BM1684X_504 python datasets linaro@<SOC_IP>:~/sam3/
```

SoC 上运行：
```bash
cd python
python3 sam3_infer.py --model_dir ../models/BM1684X_504 --precision f16 --text "a dog" --dev_id 0
```

SoC 环境要求：

| 组件 | 最低版本 | 说明 |
|------|---------|------|
| libsophon | 0.5.3+ | Sophon 基础驱动和运行时 |
| sophon-sail | 3.9.0+ | Sophon AI Inference Library |
| numpy | 1.24.0+ | 数值计算 |
| Python | 3.8+ | 运行环境 |

### 移植进度

| 步骤 | 内容 | 状态 |
|------|------|------|
| 01 | 模型分析 | ✅ 完成 |
| 02 | 环境搭建 (libsophon 0.5.3 + tpu_mlir) | ✅ 完成 |
| 03 | ONNX导出 (ViT + Neck + Grounding + TextEncoder) | ✅ 完成 |
| 04 | BModel编译 (gen_bmodel.sh --res 504, 9个bmodel) | ✅ 完成 |
| 05 | Python推理 (sam3_infer.py 全流水线) | ✅ 完成 |
| 06 | 精度测试 (truck/dog 0.99+, groceries ONNX固有累积翻转) | ✅ 完成 |
| 07 | 性能测试 (端到端1.9s, text encoder 14ms TPU) | ✅ 完成 |
| 08 | SoC部署 (SE7-32, 拷贝 bmodel+python) | ✅ 完成 |
| 09 | Grounding模型全链路集成 (Enc + Dec + TextEnc TPU, CPU numpy后处理) | ✅ 完成 |
| 10 | 自动化测试 (sam3_infer 端到端) | ✅ 完成 |
| 11 | 文档 | ✅ 完成 |
| 12 | Mask路径 (NumpyMaskDecoder CPU掩码, 2个bug修复+验证) | ✅ 完成 |
| 13 | INT8 1008 (ViT Part1-4 量化 + 流式加载落 SoC, Neck cos 0.89) | ✅ 本地+SE9(BM1688)+SE7-32(BM1684X)完成 |
| 14 | 1008 全流程 PCIe f16 增项 (Grounding+Text, --streaming, 出框验证) | ✅ 完成 |

### 已知限制

- **presence_logit_dec 丢弃**：TPU-MLIR Save:424 阻塞，不影响排序/框/mask，仅影响绝对置信度 ~0.12。
- **BF16 / F32 bmodel 不可用**：同 Save:424 阻塞 (si32→f32 Cast 导致 TPU_DYNAMIC→A53 hang)。
- **1008 Grounding 仅 PCIe f16**：1008×1008 全流程（含 Grounding/Text 出框）作为 PCIe f16 增项提供，需 `--streaming` 标志避免 OOM（见第 4 节）。SoC 因显存不足（Grounding Encoder O(N²) 随 grid 16× 增长 + ViT 顶 3GB）无法运行 1008 全流程。INT8 1008 仅 ViT+Neck 骨干（无 Grounding/Text），流式峰值 2.10GB 可落 SE7-32 默认 npu heap（2.36GB），全常驻峰值 5.96GB 需 `memory_edit.sh` 把 npu heap 调到 6GB（见 §7.1）；Neck 精度 cos ≈ 0.89 vs F16（不保真，见 §6.3）。F16 1008 单 Part 3.45GB 超默认 npu heap（2.36GB），PCIe 或改内存后的 SoC 可用。
- **504 框/点提示走 CPU 回退**。

### 参考

- [SAM 3 官方仓库](https://github.com/facebookresearch/sam3)
- [SAM 2 官方仓库](https://github.com/facebookresearch/segment-anything-2)
- [本项目SAM2例程](../SAM2)
- [本项目SAM例程](../SAM)
- [TPU-MLIR文档](https://github.com/sophgo/tpu-mlir)
