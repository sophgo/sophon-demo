# DINOv2

## 目录

- [DINOv2](#dinov2)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
    - [2.1 目录结构说明](#21-目录结构说明)
    - [2.2 SDK 特性](#22-sdk-特性)
  - [3. 准备模型与数据](#3-准备模型与数据)
    - [3.1 模型导出](#31-模型导出)
    - [3.2 数据准备](#32-数据准备)
    - [3.3 下载](#33-下载)
  - [4. 模型编译](#4-模型编译)
  - [5. 例程测试](#5-例程测试)
    - [5.1 Python 例程](#51-python-例程)
    - [5.2 C++ 例程](#52-c-例程)
  - [6. 精度测试](#6-精度测试)
    - [6.1 测试方法](#61-测试方法)
    - [6.2 测试结果](#62-测试结果)
  - [7. 性能测试](#7-性能测试)
    - [7.1 bmrt_test](#71-bmrt_test)
    - [7.2 程序运行性能](#72-程序运行性能)
  - [8. FAQ](#8-faq)

## 1. 简介

本例程对 [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) 的 `dinov2_vitb14_reg`（自监督预训练的通用视觉特征模型，ViT-B/14 + 4 个 register token，86M 参数）骨干网络进行移植：PyTorch 权重先导出 ONNX，再由 TPU-MLIR 编译为 BModel，使其能在 SOPHON BM1684X、BM1688、CV186X 上进行特征提取与推理测试。

**论文:** *DINOv2: Learning Robust Visual Features without Supervision*, Oquab et al., 2024 [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)

**输入口径**：与官方评估协议完全一致，即 `dinov2/data/transforms.py` 的 `make_classification_eval_transform()` 默认参数

```
Resize(256, BICUBIC) -> CenterCrop(224) -> ToTensor -> Normalize(ImageNet)
```

模型输入为固定的 `1x3x224x224`。官方公开的 ViT-B/14-reg4 基线（84.6%）就是这个口径；预训练原生分辨率 518x518 只在 high-res adaptation 与鲁棒性测试中使用，不是分类推理口径（详见 [8. FAQ](#8-faq)）。

**输出口径**：官方**线性探针特征**（linear probe feature）`[B, 1536]`

```
concat([x_norm_clstoken, mean(x_norm_patchtokens, dim=1)])
```

其中前 768 维为 CLS token，后 768 维为 256 个 patch token（排除 CLS 与 4 个 register）的均值，对齐官方 `dinov2/eval/linear.py: create_linear_input(use_n_blocks=1, use_avgpool=True)`。评估时在 CPU 侧接官方预训练的 1 层线性头 `dinov2_vitb14_reg4_linear_head.pth`（`weight[1000,1536]` + `bias[1000]`）得到 ImageNet-1k 1000 类 logits，取 Top-1。

> **本移植不改动模型**：BModel 只含骨干网络，官方线性头以权重文件形式在例程中做一次矩阵乘（Python 用 `.npz`，C++ 用 `.bin`），不参与量化、不重新训练。

## 2. 特性

### 2.1 目录结构说明

```bash
├── cpp                       # C++例程及其README
│   ├── README.md
│   └── dinov2_bmcv           # C++例程：设备侧imread解码 + BMCV前处理 + BMRT推理
├── datasets                  # 测试数据（download.sh下发，不进git）
├── docs                      # 本例程专用文档
│   └── DINOv2_Export_Guide.md
├── models                    # ONNX与BModel（download.sh下发或本地编译）
├── python                    # Python例程及其README
│   ├── README.md
│   ├── dinov2_opencv.py
│   └── dinov2_bmcv.py
├── scripts                   # 下载、模型导出、编译、自动测试等脚本
├── tools                     # 精度测试、性能比对等脚本
└── README.md                 # 本例程的中文指南
```

### 2.2 SDK 特性

* 支持 BM1684X、BM1688、CV186X 三个平台
* 支持 FP32、FP16、INT8 三种精度
* 支持基于 OpenCV 和 BMCV 预处理的 Python 推理（输入为图片目录）
* 支持基于 sophon-opencv 解码 + BMCV 预处理的 C++ 推理，输入支持单张图片或图片目录（不依赖 sophon-ffmpeg）
* 输出 1536 维特征与 Top-1 分类结果

> **实测范围**：本文档的全部精度与性能数据均来自 SoC 盒子（SE7-32 / SE9-16 / SE9-8）。x86/arm/riscv PCIe 平台的编译与运行方式沿用仓库通用文档，本版本**未在 PCIe 卡上实测**；C++ 的 PCIe 产物（`dinov2_bmcv.pcie`）与 SoC 产物（`dinov2_bmcv.soc`）来自同一份源码，仅目标架构不同。

## 3. 准备模型与数据

建议使用 TPU-MLIR 编译 BModel，PyTorch 模型在编译前需导出为 ONNX。完整的导出原理、位置编码烘焙与自检流程见 [docs/DINOv2_Export_Guide.md](docs/DINOv2_Export_Guide.md)。

### 3.1 模型导出

```bash
# 需要 dinov2 源码（facebookresearch/dinov2）与 Meta 分发的预训练权重（下载方式见导出指南 §1.4）
python3 scripts/export_onnx.py \
    --dinov2_src <path/to/dinov2_repo> \
    --weights models/torch/dinov2_vitb14_reg4_pretrain.pth \
    --out_onnx models/onnx/dinov2_vitb14_reg_224_1b.onnx

# 官方线性头转成 numpy / 二进制格式（例程推理用）
python3 scripts/export_linear_head.py \
    --weights models/torch/dinov2_vitb14_reg4_linear_head.pth \
    --out datasets/dinov2_vitb14_reg4_linear_head.npz \
    --bin_out datasets/dinov2_vitb14_reg4_linear_head.bin
```

> **导出环境**：需要 Python 3.10 + `torch>=2.6`（`dynamo=True` 导出路径）+ `onnx/onnxsim/onnxruntime`，以及 clone 下来的官方 dinov2 源码（仅通过 `--dinov2_src` 引用，不要安装其 `requirements.txt`）。**不要在 TPU-MLIR 容器默认环境（torch 2.0/2.1）中直接导出**。完整安装命令、验证版本自检与各脚本依赖范围见 [导出指南 §1](docs/DINOv2_Export_Guide.md#1-准备工作)，依赖清单见 `scripts/requirements-export.txt`。

> **权重许可**：DINOv2 原始 PyTorch 权重（骨干与官方线性头）按 Meta 的 [DINOv2 License](https://github.com/facebookresearch/dinov2/blob/main/LICENSE) 分发，**不随本仓库提供**，需自行下载（官方仓库 README 提供直链，见 [导出指南 §1.4](docs/DINOv2_Export_Guide.md#14-获取预训练权重meta-license)）。`scripts/download.sh` 提供的是已导出的 ONNX、编译好的 BModel、测试数据集，以及由官方线性头转换出的 `.npz/.bin`。

### 3.2 数据准备

* **ImageNet-1k 验证集**：官方 `ILSVRC2012` 验证集，评估用其 `imagenet_val_1k` 千张子集（每类 1 张，`label.txt` + 平铺 `img/` 目录，格式为 `文件名\t类别下标`），与 ResNet 例程口径一致。
* **量化标定集**：`datasets/cali_data`（INT8 标定用，200 张，按官方 eval 前处理生成）。
* **编译校验输入**：`datasets/test_input/dinov2_1b_in_f32.npz`（`gen_*bmodel_mlir.sh` 的 `model_transform.py` / `model_deploy.py` 用它做数值校验）。
* **线性头**：`datasets/dinov2_vitb14_reg4_linear_head.npz`（Python）与 `.bin`（C++）。
* **类别映射**：`datasets/imagenet_synset_to_idx.json`（`gen_val_1k.py` 生成验证集时使用）与 `datasets/imagenet_classes.txt`（例程打印类别名）。

上述数据均随 `datasets.zip` 提供。**如果走“自行重新生成”路径，请先执行 [3.3 下载](#33-下载) 取得 `imagenet_synset_to_idx.json`**（该映射不在仓库内），再执行（均按官方 eval 前处理）：

```bash
# --val_zip 为按 synset 文件夹组织的 ImageNet val zip
python3 scripts/gen_val_1k.py --val_zip <imagenet_validation.zip>
python3 scripts/gen_cali_data.py --val_zip <imagenet_validation.zip> --num 200
```

> 注意：官方发布的 `ILSVRC2012_img_val.tar` 是**扁平结构**（所有图片在根目录、无类别文件夹，且包内无标签），只能直接用于 `gen_cali_data.py`；`gen_val_1k.py` 依赖“synset 文件夹名”生成 `label.txt`，需先用 devkit 的 `ILSVRC2012_validation_ground_truth.txt` + `synsets.txt` 把 val 整理成 `n01440764/xxx.JPEG` 形式，或直接使用带标签的 ImageNet-1k val 数据集。

### 3.3 下载

```bash
bash scripts/download.sh          # 首次下载；已存在的目录会跳过
bash scripts/download.sh --force  # 强制重新下载并覆盖（升级版本时用）
```

> 依赖 `unzip` 和 `python3`；缺少 `dfss` 时脚本会自动 `pip3 install`。SoC 盒子默认可能没有 `unzip`，可先在开发机上执行下载再把 `models/`、`datasets/` 拷贝到盒子。

从 dfss 下载 `datasets.zip`（数据集 + 标定集 + `test_input` + 线性头）、`onnx_224.zip`（ONNX）与 `BM1684X.zip` / `BM1688.zip` / `CV186X.zip`（各芯片预编译 BModel）。脚本会检查关键文件是否齐全，缺件时自动重新下载。

## 4. 模型编译

直接使用 `scripts/download.sh` 下发的预编译 BModel 可**跳过本节**；只有自行编译才需要 TPU-MLIR 环境（安装见 [Environment_Install_Guide](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)）。

在 TPU-MLIR 开发容器内进入 `scripts` 目录执行。本例程目前只适配并验证了 `bm1684x` / `bm1688` / `cv186x` 三种芯片（对应 SE7-32 / SE9-16 / SE9-8）；SE5-16(BM1684) 未做适配。

> **工具链版本（重要）**：本文档所有数据均由 **TPU-MLIR `v1.30.beta.0-46-gb2e0030-20261029`**（whl `tpu_mlir-1.30b0`，2026-10-29 上游 HEAD）编译。该版本包含 BM1686（BM1688 / CV186X）dequant/requant 的连续 staging 修复（上游 commit `bb0cb8e`），把混合精度图中 48 处 INT8→F16 反量化的逐元素 gather 改为连续 2D 搬运，**INT8 在这两颗芯片上不再慢于 FP16**（详见 [8. FAQ](#8-faq) 第 4 条）。修复在编译期烤进 bmodel，**盒子无需刷固件/升级驱动**。
>
> 用旧版 TPU-MLIR 编译出的 INT8 bmodel 仍是旧表现；要获得新性能必须**重新编译**（不能只换推理库）。同理，重新编译 INT8 时请使用该版本或更新版本。

```bash
cd scripts
./gen_fp32bmodel_mlir.sh bm1684x
./gen_fp16bmodel_mlir.sh bm1684x
./gen_int8bmodel_mlir.sh bm1684x
# bm1688 / cv186x 同理，参数换成对应芯片名
```

下载与编译产物：

```
models
├── onnx
│   └── dinov2_vitb14_reg_224_1b.onnx   （download.sh 提供；自行导出时由 3.1 生成）
├── BM1684X
│   ├── dinov2_vitb14_reg_fp32_1b.bmodel
│   ├── dinov2_vitb14_reg_fp16_1b.bmodel
│   └── dinov2_vitb14_reg_int8_1b.bmodel
├── BM1688   （fp32/fp16/int8 另含 *_1b_2core.bmodel）
└── CV186X   （单核芯片，无 2core）
```

> **INT8 量化配方（关键）**：DINOv2 的 linear-probe 特征对量化极敏感，**全 INT8 会直接毁掉特征**（SE9-16 实测 Top-1 仅 0.1%，与随机相当）。正确配方是「只把 12 个 block 的 48 个带 bias 线性层（每 block 4 个：qkv 融合投影 `768→2304`、`attn.proj` `768→768`、`mlp.fc1` `768→3072`、`mlp.fc2` `3072→768`）留在 INT8，其余算子（24 个无 bias 的 attention 矩阵乘 `QK^T`/`attn@V`、patch 均值、LayerNorm / Softmax / GELU / 残差 / 布局）全部 F16」——`run_calibration.py --sq --cali_method mse`（200 张标定图）+ `--quantize_table`（`scripts/gen_qtable.py` 生成的白名单）+ `--matmul_perchannel`。详见 `scripts/gen_int8bmodel_mlir.sh` 顶部注释。
>
> **精度档统一**：三颗芯片都只提供 FP32 / FP16 / INT8（48 个线性层 INT8 + 其余 F16）三种档位，不生成 tail16/tail24 之类的混合精度变体。INT8 在三个平台都最快（见 [7.1](#71-bmrt_test)），代价是 Top-1 低 0.8~1.9 个点；要求精度与 FP16 对齐时直接用 FP16。
>
> **关于 INT8 的数值校验门限**：`model_deploy.py` 对 INT8 的默认门限是 `0.90,0.80`（cosine, euclidean），按分类 logits 标定；本模型比对的是 1536 维原始特征，逐元素噪声分散在整个向量上，三平台实测为 cosine 0.948~0.951 / euclidean 0.673~0.682（bm1684x 0.9507/0.6819、bm1688 与 cv186x 均为 0.9479/0.6725），默认门限下必然判 FAILED。脚本中显式设为 `--tolerance 0.90,0.60`（保留校验，不跳过），对应的真实精度损失是 Top-1 −0.8~1.9 个点（见 [6.2](#62-测试结果)）。同仓库先例：SuperGlue int8 用 `--tolerance 0.5,0`、CenterNet int8 用 `--tolerance 0.8,0.5`。

## 5. 例程测试

详细说明见 [C++例程](cpp/README.md) 与 [Python例程](python/README.md)。运行环境（libsophon、sophon-opencv、sophon-ffmpeg、sophon-sail 与 TPU-MLIR；其中 sophon-ffmpeg 仅供 Python 的 SAIL 解码使用，C++ 例程不依赖）的安装见 [Environment_Install_Guide](../../docs/Environment_Install_Guide.md) 与 [python/README.md](python/README.md) 第 1 节。

### 5.1 Python 例程

```bash
cd python
# OpenCV 解码 + OpenCV 前处理 + SAIL 推理
python3 dinov2_opencv.py --input ../datasets/imagenet_val_1k/img \
    --bmodel ../models/BM1688/dinov2_vitb14_reg_fp16_1b.bmodel \
    --head ../datasets/dinov2_vitb14_reg4_linear_head.npz --dev_id 0

# SAIL 解码 + BMCV 前处理 + SAIL 推理
python3 dinov2_bmcv.py --input ../datasets/imagenet_val_1k/img \
    --bmodel ../models/BM1688/dinov2_vitb14_reg_fp16_1b.bmodel \
    --head ../datasets/dinov2_vitb14_reg4_linear_head.npz --dev_id 0
```

> 上面以 SE9-16 的 `models/BM1688/` 为例；在其他平台请把模型目录换成 `models/BM1684X/` 或 `models/CV186X/`，文件名规则相同。SE9-16 还可以直接用双核模型 `dinov2_vitb14_reg_{fp32,fp16,int8}_1b_2core.bmodel`，用法与 1core 完全相同。若模型在别处，`--bmodel` 传绝对路径即可。

### 5.2 C++ 例程

见 [cpp/README.md](cpp/README.md)。SoC 模式下编译产物为 `dinov2_bmcv.soc`。

## 6. 精度测试

### 6.1 测试方法

在 `datasets/imagenet_val_1k`（每类 1 张共 1000 张）上跑完整推理：骨干输出 `[B,1536]` 特征后，在 CPU 侧接官方 1 层线性头得到 1000 类 logits，与 `label.txt` 比对计算 Top-1。

```bash
# 例程推理会写出 results/*.json（含每张图的 prediction）
python3 python/dinov2_bmcv.py --input datasets/imagenet_val_1k/img \
    --bmodel models/BM1688/dinov2_vitb14_reg_fp16_1b.bmodel \
    --head datasets/dinov2_vitb14_reg4_linear_head.npz \
    --classnames datasets/imagenet_classes.txt --dev_id 0
python3 tools/eval_imagenet.py --gt_path datasets/imagenet_val_1k/label.txt \
    --result_json results/dinov2_vitb14_reg_fp16_1b.bmodel_img_bmcv_python_result.json
```

> 同样以 SE9-16 的 `models/BM1688/` 为例，其他平台把 `--bmodel` 换成 `models/BM1684X/` 或 `models/CV186X/` 下的同名文件；`eval_imagenet.py` 的 `--result_json` 要与实际生成的文件名一致（文件名由 bmodel 名 + 输入目录名拼成）。

完整自动测试入口见 `scripts/auto_test.sh`（`-m` 选择编译 / PCIe 构建 / SoC 构建 / 测试，`-p pytest` 只表示把结果写入 `auto_test_result.txt`，不是 pytest 框架）。

### 6.2 测试结果

精度参考点（同一套 1000 张子集、同一个官方线性头）：

| 路径 | 前处理 | Top-1(%) | 说明 |
| -------- | -------- | -------- | -------- |
| 官方 MODEL_CARD | 官方 full val | 84.6 | `dinov2_vitb14_reg` ImageNet-1k linear probe 公开指标 |
| ONNX / CPU | PIL BICUBIC（官方前处理） | 84.30 | 本仓库 1k 子集上按官方 eval 协议实测，作为端口对齐基线 |
| bmodel / bmcv（FP32/FP16） | VPP 硬件缩放 | 84.10 | 本例程 FP 路径实测 |
| bmodel / opencv（FP32/FP16） | cv2 BICUBIC | 83.90 | 本例程 FP 路径实测 |

上表可见端口实现与官方 84.6 的差距为 0.3~0.7 个点（ONNX/CPU 差 0.3、bmcv 差 0.5、opencv 差 0.7）；相对本例程的 ONNX/CPU 基线 84.30 则仅 0.2~0.4 个点。差距来自重采样内核（PIL / VPP / cv2）、bmodel 数值精度与 1k 子集抽样方差，属正常范围。

> **注：官方基线有两个数字，都对。** 官方 MODEL_CARD / 模型表里 `dinov2_vitb14_reg`（带 registers）的 ImageNet-1k **linear** 是 **84.6%**（不带 registers 的 `dinov2_vitb14` 是 84.5%）；而仓库 README 另一张“已发布 1 层线性头”表对 `dinov2_vitb14_reg4_linear_head.pth`（即本例程使用的线性头）记 **84.4%**。两条都是官方评估记录，差 0.2 点是评估流程细节不同，不是模型或本端口的问题。本表采用 84.6 作为模型级基线。

各平台完整精度结果（1000 张）。同一 bmodel 在不同例程间的 Top-1 差异：FP32/FP16 路径 ≤0.3 个点（1000 张即 ≤3 张图）；INT8 路径可达 0.5~1.3 个点（量化会放大不同解码/缩放内核带来的微小像素差异，如 SE7-32 标准 INT8 opencv 82.00 vs bmcv 83.30）。差异来自前处理/解码路径（VPP 硬件缩放 vs cv2 BICUBIC、SAIL 解码 vs sophon-opencv 设备侧解码），非模型差异。

**SE9-16（BM1688）**

| 测试程序 | 测试模型 | Top-1(%) |
| -------- | -------- | -------- |
| dinov2_opencv.py | dinov2_vitb14_reg_fp32_1b.bmodel | 83.90 |
| dinov2_opencv.py | dinov2_vitb14_reg_fp32_1b_2core.bmodel | 83.90 |
| dinov2_opencv.py | dinov2_vitb14_reg_fp16_1b.bmodel | 83.90 |
| dinov2_opencv.py | dinov2_vitb14_reg_fp16_1b_2core.bmodel | 83.90 |
| dinov2_opencv.py | dinov2_vitb14_reg_int8_1b.bmodel | 82.40 |
| dinov2_opencv.py | dinov2_vitb14_reg_int8_1b_2core.bmodel | 82.40 |
| dinov2_bmcv.py | dinov2_vitb14_reg_fp32_1b.bmodel | 84.10 |
| dinov2_bmcv.py | dinov2_vitb14_reg_fp32_1b_2core.bmodel | 84.10 |
| dinov2_bmcv.py | dinov2_vitb14_reg_fp16_1b.bmodel | 84.10 |
| dinov2_bmcv.py | dinov2_vitb14_reg_fp16_1b_2core.bmodel | 84.10 |
| dinov2_bmcv.py | dinov2_vitb14_reg_int8_1b.bmodel | 82.90 |
| dinov2_bmcv.py | dinov2_vitb14_reg_int8_1b_2core.bmodel | 82.90 |
| dinov2_bmcv.soc | dinov2_vitb14_reg_fp32_1b.bmodel | 83.90 |
| dinov2_bmcv.soc | dinov2_vitb14_reg_fp32_1b_2core.bmodel | 83.90 |
| dinov2_bmcv.soc | dinov2_vitb14_reg_fp16_1b.bmodel | 83.90 |
| dinov2_bmcv.soc | dinov2_vitb14_reg_fp16_1b_2core.bmodel | 83.90 |
| dinov2_bmcv.soc | dinov2_vitb14_reg_int8_1b.bmodel | 82.20 |
| dinov2_bmcv.soc | dinov2_vitb14_reg_int8_1b_2core.bmodel | 82.20 |

**SE9-8（CV186X）**

| 测试程序 | 测试模型 | Top-1(%) |
| -------- | -------- | -------- |
| dinov2_opencv.py | dinov2_vitb14_reg_fp32_1b.bmodel | 83.90 |
| dinov2_opencv.py | dinov2_vitb14_reg_fp16_1b.bmodel | 83.90 |
| dinov2_opencv.py | dinov2_vitb14_reg_int8_1b.bmodel | 82.40 |
| dinov2_bmcv.py | dinov2_vitb14_reg_fp32_1b.bmodel | 84.10 |
| dinov2_bmcv.py | dinov2_vitb14_reg_fp16_1b.bmodel | 84.10 |
| dinov2_bmcv.py | dinov2_vitb14_reg_int8_1b.bmodel | 82.90 |
| dinov2_bmcv.soc | dinov2_vitb14_reg_fp32_1b.bmodel | 83.90 |
| dinov2_bmcv.soc | dinov2_vitb14_reg_fp16_1b.bmodel | 83.90 |
| dinov2_bmcv.soc | dinov2_vitb14_reg_int8_1b.bmodel | 82.20 |

**SE7-32（BM1684X）**

| 测试程序 | 测试模型 | Top-1(%) |
| -------- | -------- | -------- |
| dinov2_opencv.py | dinov2_vitb14_reg_fp32_1b.bmodel | 83.90 |
| dinov2_opencv.py | dinov2_vitb14_reg_fp16_1b.bmodel | 83.90 |
| dinov2_opencv.py | dinov2_vitb14_reg_int8_1b.bmodel | 82.00 |
| dinov2_bmcv.py | dinov2_vitb14_reg_fp32_1b.bmodel | 84.10 |
| dinov2_bmcv.py | dinov2_vitb14_reg_fp16_1b.bmodel | 84.10 |
| dinov2_bmcv.py | dinov2_vitb14_reg_int8_1b.bmodel | 83.30 |
| dinov2_bmcv.soc | dinov2_vitb14_reg_fp32_1b.bmodel | 84.00 |
| dinov2_bmcv.soc | dinov2_vitb14_reg_fp16_1b.bmodel | 84.10 |
| dinov2_bmcv.soc | dinov2_vitb14_reg_int8_1b.bmodel | 82.70 |

> **测试说明**：
>
> 1. FP32 与 FP16 的 Top-1 基本完全一致（唯一差异是 BM1684X C++ 例程 84.00 vs 84.10，1k 张里差 1 张），1b 与 2core 版本也完全一致，说明本例程在 FP16 下已无数值损失（官方口径本身也只有 84.6），且双核切分不影响结果。
> 2. 标准 INT8 相比同例程的 FP16 损失 0.8~1.9 个点（BM1688 bmcv 82.90 vs 84.10、BM1684X bmcv 83.30 vs 84.10、BM1684X opencv 82.00 vs 83.90、CV186X bmcv 82.90 vs 84.10），与其它例程的 INT8 损失同级。
> 3. C++ 例程在三个平台都实测了各自的全部模型（FP32/FP16/INT8，见上表 `dinov2_bmcv.soc` 各行）。与同平台 Python bmcv 相比：FP32/FP16 差异 ≤0.2 点（来自解码/VPP 与 SAIL 解码/缩放路径差异）；INT8 差异不超过 0.7 点（SE9-16 82.20 vs 82.90、CV186X 82.20 vs 82.90、BM1684X 82.70 vs 83.30）。

## 7. 性能测试

### 7.1 bmrt_test

使用 `bmrt_test` 测试模型的理论推理时间：

```bash
# bmrt_test 在 SoC 上通常不在 PATH 里，用 libsophon 的绝对路径调用
/opt/sophon/libsophon-current/bin/bmrt_test --bmodel models/BM1688/dinov2_vitb14_reg_fp16_1b.bmodel --devid 0
```

| 测试平台 | 测试模型 | calculate time(ms) |
| -------- | -------- | ------------------ |
| SE9-16 | BM1688/dinov2_vitb14_reg_fp32_1b.bmodel | 293.47 |
| SE9-16 | BM1688/dinov2_vitb14_reg_fp32_1b_2core.bmodel | 193.96 |
| SE9-16 | BM1688/dinov2_vitb14_reg_fp16_1b.bmodel | 52.75 |
| SE9-16 | BM1688/dinov2_vitb14_reg_fp16_1b_2core.bmodel | 40.77 |
| SE9-16 | BM1688/dinov2_vitb14_reg_int8_1b.bmodel | 33.89 |
| SE9-16 | BM1688/dinov2_vitb14_reg_int8_1b_2core.bmodel | 28.23 |
| SE9-8 | CV186X/dinov2_vitb14_reg_fp32_1b.bmodel | 293.38 |
| SE9-8 | CV186X/dinov2_vitb14_reg_fp16_1b.bmodel | 52.74 |
| SE9-8 | CV186X/dinov2_vitb14_reg_int8_1b.bmodel | 33.88 |
| SE7-32 | BM1684X/dinov2_vitb14_reg_fp32_1b.bmodel | 81.31 |
| SE7-32 | BM1684X/dinov2_vitb14_reg_fp16_1b.bmodel | 18.47 |
| SE7-32 | BM1684X/dinov2_vitb14_reg_int8_1b.bmodel | 16.95 |

> **说明**：FP32 在 BM1688/CV186X 上没有硬件加速，延迟约为 FP16 的 5.6 倍（同一量级的差距在其它例程上也存在，如 ResNet50 在 SE9-16 上 fp32 43.42ms vs fp16 7.19ms），BM1684X 上约为 4.4 倍；故 FP32 主要用作数值基准。
>
> 使用新版工具链（见第 4 节）编译后，**INT8 在三颗芯片上都快于 FP16**：BM1688 单核 33.89 vs 52.75ms（1.56×）、双核 28.23 vs 40.77ms（1.44×），CV186X 33.88 vs 52.74ms（1.56×），BM1684X 16.95 vs 18.47ms（1.09×）。旧版 TPU-MLIR 编译的 INT8 在 BM1688/CV186X 上曾慢于 FP16（140.64ms），原因与修复见 [8. FAQ](#8-faq) 第 4 条。
>
> **测试环境**：SE7-32（libsophon 0.5.3）、SE9-16 / SE9-8（libsophon 0.4.14）。bmodel 直接运行，不需要额外刷固件或升级驱动；换用其它 SDK 版本时延迟可能有小幅差异。

### 7.2 程序运行性能

参考 [C++例程](cpp/README.md) 或 [Python例程](python/README.md) 运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++ 和 Python 例程打印的时间已经折算为单张图片的处理时间。

下表为在三个平台上使用不同例程、不同模型测试 `datasets/imagenet_val_1k` 的结果。各例程统计的解码/预处理/推理/后处理时间，均已折算为单张图片时间（ms）。`inference_time` 为完整一次推理调用（Python 的 `engine.process()` / C++ 的 `DINOV2::Classify()`）的墙钟耗时，含设备与主机间的数据搬运，故略高于 bmrt_test 的纯 NPU `calculate time`。

**SE9-16（BM1688）**

| 测试程序 | 测试模型 | decode_time | preprocess_time | inference_time | postprocess_time |
| -------- | -------- | ----------- | --------------- | -------------- | ---------------- |
| dinov2_opencv.py | dinov2_vitb14_reg_fp32_1b.bmodel | 7.68 | 12.35 | 294.50 | 6.00 |
| dinov2_opencv.py | dinov2_vitb14_reg_fp32_1b_2core.bmodel | 7.63 | 12.36 | 194.93 | 5.93 |
| dinov2_opencv.py | dinov2_vitb14_reg_fp16_1b.bmodel | 7.62 | 12.35 | 53.71 | 5.95 |
| dinov2_opencv.py | dinov2_vitb14_reg_fp16_1b_2core.bmodel | 7.65 | 12.36 | 41.72 | 5.94 |
| dinov2_opencv.py | dinov2_vitb14_reg_int8_1b.bmodel | 7.64 | 12.33 | 34.85 | 5.95 |
| dinov2_opencv.py | dinov2_vitb14_reg_int8_1b_2core.bmodel | 7.66 | 12.36 | 29.18 | 5.93 |
| dinov2_bmcv.py | dinov2_vitb14_reg_fp32_1b.bmodel | 6.89\* | 1.67 | 294.01 | 5.91 |
| dinov2_bmcv.py | dinov2_vitb14_reg_fp32_1b_2core.bmodel | 3.53 | 1.65 | 194.57 | 5.97 |
| dinov2_bmcv.py | dinov2_vitb14_reg_fp16_1b.bmodel | 3.45 | 1.66 | 53.19 | 5.96 |
| dinov2_bmcv.py | dinov2_vitb14_reg_fp16_1b_2core.bmodel | 3.45 | 1.64 | 41.18 | 5.90 |
| dinov2_bmcv.py | dinov2_vitb14_reg_int8_1b.bmodel | 3.43 | 1.66 | 34.31 | 5.89 |
| dinov2_bmcv.py | dinov2_vitb14_reg_int8_1b_2core.bmodel | 3.44 | 1.65 | 28.65 | 5.89 |
| dinov2_bmcv.soc | dinov2_vitb14_reg_fp32_1b.bmodel | 3.229 | 0.852 | 293.535 | 4.058 |
| dinov2_bmcv.soc | dinov2_vitb14_reg_fp32_1b_2core.bmodel | 3.216 | 0.855 | 193.962 | 4.041 |
| dinov2_bmcv.soc | dinov2_vitb14_reg_fp16_1b.bmodel | 3.218 | 0.850 | 52.786 | 4.040 |
| dinov2_bmcv.soc | dinov2_vitb14_reg_fp16_1b_2core.bmodel | 3.188 | 0.851 | 40.797 | 4.030 |
| dinov2_bmcv.soc | dinov2_vitb14_reg_int8_1b.bmodel | 3.210 | 0.844 | 33.908 | 4.029 |
| dinov2_bmcv.soc | dinov2_vitb14_reg_int8_1b_2core.bmodel | 3.183 | 0.848 | 28.260 | 4.043 |

**SE9-8（CV186X）**

| 测试程序 | 测试模型 | decode_time | preprocess_time | inference_time | postprocess_time |
| -------- | -------- | ----------- | --------------- | -------------- | ---------------- |
| dinov2_opencv.py | dinov2_vitb14_reg_fp32_1b.bmodel | 8.69 | 13.28 | 294.38 | 1.75 |
| dinov2_opencv.py | dinov2_vitb14_reg_fp16_1b.bmodel | 8.99 | 13.61 | 54.00 | 2.38 |
| dinov2_opencv.py | dinov2_vitb14_reg_int8_1b.bmodel | 8.79 | 13.60 | 35.10 | 1.97 |
| dinov2_bmcv.py | dinov2_vitb14_reg_fp32_1b.bmodel | 5.44 | 1.78 | 293.83 | 1.79 |
| dinov2_bmcv.py | dinov2_vitb14_reg_fp16_1b.bmodel | 4.24 | 1.83 | 53.59 | 2.16 |
| dinov2_bmcv.py | dinov2_vitb14_reg_int8_1b.bmodel | 4.17 | 1.89 | 34.48 | 1.89 |
| dinov2_bmcv.soc | dinov2_vitb14_reg_fp32_1b.bmodel | 3.363 | 0.859 | 293.319 | 4.100 |
| dinov2_bmcv.soc | dinov2_vitb14_reg_fp16_1b.bmodel | 3.187 | 0.855 | 52.654 | 4.091 |
| dinov2_bmcv.soc | dinov2_vitb14_reg_int8_1b.bmodel | 3.157 | 0.862 | 33.808 | 4.068 |

**SE7-32（BM1684X）**

| 测试程序 | 测试模型 | decode_time | preprocess_time | inference_time | postprocess_time |
| -------- | -------- | ----------- | --------------- | -------------- | ---------------- |
| dinov2_opencv.py | dinov2_vitb14_reg_fp32_1b.bmodel | 5.89 | 9.28 | 81.90 | 2.24 |
| dinov2_opencv.py | dinov2_vitb14_reg_fp16_1b.bmodel | 5.83 | 9.07 | 19.11 | 1.85 |
| dinov2_opencv.py | dinov2_vitb14_reg_int8_1b.bmodel | 5.82 | 9.17 | 17.54 | 1.86 |
| dinov2_bmcv.py | dinov2_vitb14_reg_fp32_1b.bmodel | 2.83 | 0.77 | 81.45 | 2.06 |
| dinov2_bmcv.py | dinov2_vitb14_reg_fp16_1b.bmodel | 2.64 | 0.76 | 18.69 | 1.94 |
| dinov2_bmcv.py | dinov2_vitb14_reg_int8_1b.bmodel | 2.87 | 0.87 | 17.17 | 2.38 |
| dinov2_bmcv.soc | dinov2_vitb14_reg_fp32_1b.bmodel | 2.365 | 0.350 | 81.175 | 3.265 |
| dinov2_bmcv.soc | dinov2_vitb14_reg_fp16_1b.bmodel | 2.362 | 0.345 | 18.375 | 3.269 |
| dinov2_bmcv.soc | dinov2_vitb14_reg_int8_1b.bmodel | 2.353 | 0.344 | 16.788 | 3.277 |

> **说明**：
>
> 1. 3 个例程的后处理逻辑一致：线性头 `1000x1536` 矩阵乘加 + softmax + topk，约为 1.8~6ms；CV186X 为单核 A53，后处理约 2ms。
> 2. bmcv 例程的预处理走 VPP 硬件（SE9 平台约 1.6~1.9ms，SE7-32 约 0.8ms），C++ 例程同为 VPP（SE9 约 0.85ms，SE7 约 0.35ms）；opencv 例程为 CPU 缩放（约 9~13.6ms），是其预处理耗时的主要来源。
> 3. 解码：Python bmcv 例程用 SAIL 硬件解码（约 2.6~5.5ms），opencv 例程用 `cv2.imread`（约 5.8~9ms），C++ 例程用 sophon-opencv 设备侧 `cv::imread(..., dev_id)` + `cv::bmcv::toBMI`（SE9 约 3.2~3.4ms，SE7-32 约 2.4ms）。首次运行包含 JPU/图片缓存冷启动开销，会略高。
> 4. `*` 为该轮测试的第一个模型，SAIL 解码首次调用含冷启动开销；后续模型均稳定在约 3.5ms。

## 8. FAQ

**1. 为什么输入是 224x224 而不是 518x518？**

官方评估协议 `dinov2/data/transforms.py: make_classification_eval_transform()` 的默认参数是 `Resize(256, BICUBIC)` + `CenterCrop(224)`，模型输入 224，官方公开的 ViT-B/14-reg4 指标（84.6%）就是这个口径。518 是预训练原生分辨率，只在 high-res adaptation 训练与 MODEL_CARD 的鲁棒性测试中出现，不是分类推理口径。本例程与官方 eval 保持完全一致，便于横向对比。

**2. 为什么用线性探针（linear probe）而不是直接分类？**

DINOv2 是自监督模型，骨干输出的是通用特征，没有内置分类头。官方用「冻结骨干 + 1 层线性头」在 ImageNet 上评估迁移能力，本例程对齐该协议：骨干输出 1536 维特征（`concat(CLS, mean(patch))`），CPU 侧接官方预训练线性头算 Top-1。线性头不编入 BModel，也不做任何再训练。

**3. 为什么 INT8 必须用混合精度？**

DINOv2 的特征对量化误差极其敏感：LayerNorm/GELU/Softmax 的输出动态范围大，全 INT8 会把特征方向毁掉。SE9-16 实测全 INT8 的 Top-1 仅 **0.1%**（与随机相当，且已验证不是输出 dtype 的问题——把输出层单独改成 F16 后仍为 0.2%）。因此必须保留 LayerNorm/Softmax/GELU/残差为 F16，只量化线性层。

**4. 为什么旧版工具链下 INT8 比 FP16 慢，现在又快回来了？**

本模型是混合精度（48 个线性层 INT8 + 其余 F16）。48 个 INT8 矩阵乘的输出要反量化回 F16 交给下游的 LayerNorm/Softmax/GELU/残差，图上因此存在 48 处 INT8→F16 dequant（以及对应的 requant）。在 BM1686（BM1688 / CV186X）后端上，**旧版 TPU-MLIR 把这些 dequant/requant 降成了逐元素 gather**：每次只搬一个元素，DMA 描述符爆炸，单是这一段就消耗约 114M 个 TDMA cycle。于是同一块 SE9-16、同一份混合精度图，bmrt 延迟从纯 FP16 的 52.81ms（旧工具链实测；新工具链 52.75ms，见 [7.1](#71-bmrt_test)）被推到 **140.64ms**，比把整张图做成全 INT8（92.24ms）还慢——这不是量化配方的问题，是反量化 lowering 的实现缺陷。

修复已合入 **TPU-MLIR `v1.30.beta.0-46-gb2e0030-20261029`**（上游 commit `bb0cb8e`，BM1686 dequant/requant 连续 staging）：非 perchannel 的 dequant/requant 改为连续 2D 搬运，TDMA cycle 从 **114M 降到 17.6M**，反量化开销基本消失。修复在**编译期**烤进 bmodel，盒子无需刷固件/升级驱动，但必须用该版本或更新版本**重新编译** INT8 模型。新工具链实测：

| 芯片 | FP16 bmrt(ms) 1c / 2c | INT8 bmrt(ms) 1c / 2c | INT8 加速比 |
| -------- | -------- | -------- | -------- |
| BM1688（SE9-16） | 52.75 / 40.77 | **33.89 / 28.23** | 1.56× / 1.44× |
| CV186X（SE9-8） | 52.74 / — | **33.88 / —** | 1.56× |
| BM1684X（SE7-32） | 18.47 / — | **16.95 / —** | 1.09× |

BM1684X 上旧版路径本来就正常（per-channel dequant 走的是高效分支），所以新旧工具链差别很小（16.82 → 16.95ms），但 INT8 仍是该芯片上最快的精度。

**INT8 的代价是精度（−0.8~1.9 个点）。** 三颗芯片统一使用同一份配方（48 个投影 Gemm 全 INT8），不提供 tail 混合精度档。部署建议：

* **BM1688**：最低延迟用双核 `int8_1b_2core`（28.65ms / 82.90%）；要求精度与 FP16 对齐用 `fp16_1b_2core`（41.18ms / 84.10%）。
* **CV186X**：单核 `int8_1b`（34.48ms / 82.90%）；精度优先用 `fp16_1b`（53.59ms / 84.10%）。
* **BM1684X**：`int8_1b`（17.17ms / 83.30%）只比 FP16（18.69ms / 84.10%）低 0.8 点，延迟优先选 INT8、精度优先选 FP16。

> 已实测排除的无效旋钮：`--asymmetric`（Top-1 崩到 0.1%，输出坍缩为常数）、`--opt 3`（tpuc-opt 在 layer-group 阶段 abort）、`--opt 1`（与默认 `--opt 2` 无差异）。`--num_core 2` 始终有效且不影响精度。
>
> 体积方面：标准 INT8 94MB，小于 FP16 178MB / FP32 350MB。新版工具链下 INT8 在全部三颗芯片上都是「更快 + 更小」。

**5. 为什么 FP32 在边缘设备上这么慢？**

FP32 在 BM1688/CV186X 上没有硬件加速，延迟约为 FP16 的 5.6 倍（本例程实测 bmrt_test 293.47ms vs 52.75ms；BM1684X 上约 4.4 倍，81.31ms vs 18.47ms）。同仓库其它例程也是同一量级的差距，例如 ResNet50 在 SE9-16 上是 fp32 43.42ms / fp16 7.19ms。因此 FP32 主要作为数值基准，实际部署建议使用 FP16 或 INT8。

**6. 如何下载模型和数据集？**

```bash
bash scripts/download.sh          # 首次下载，已完整的目录自动跳过
bash scripts/download.sh --force  # 强制重新下载并覆盖
```

脚本从 dfss 拉取 `datasets.zip`（1000 张验证子集、200 张标定集、`test_input`、官方线性头 `.npz/.bin`）、`onnx_224.zip` 与 `BM1684X.zip` / `BM1688.zip` / `CV186X.zip`（各芯片预编译 bmodel，BM1688 另含 2core）。原始 PyTorch 权重（`.pth`）按 Meta DINOv2 License 分发，**不随包提供**；只有自行导出 ONNX 时才需要（见 3.1）。依赖 `unzip` 与 `dfss`（缺失时脚本会自动 `pip3 install`）。

**7. 如何编译 BModel？**

在 TPU-MLIR 开发容器内进入 `scripts/` 按芯片执行（ONNX 由 `download.sh` 提供；若要自行导出，先在**独立 Python 环境**按 [导出指南 §1](docs/DINOv2_Export_Guide.md#1-准备工作) 安装 `torch>=2.6` 等依赖并执行 3.1，不要用 TPU-MLIR 容器自带的 torch 2.0/2.1 导出）：

```bash
./gen_fp32bmodel_mlir.sh bm1684x   # 或 bm1688 / cv186x
./gen_fp16bmodel_mlir.sh bm1684x
./gen_int8bmodel_mlir.sh bm1684x
```

INT8 需要 `datasets/cali_data` 与 `datasets/test_input`（`download.sh` 已提供）；脚本内部依次执行 `model_transform.py` → `run_calibration.py --sq --cali_method mse` → `gen_qtable.py` → `model_deploy.py`。BM1688 额外生成 2core 双核版本。**必须使用第 4 节指定的 TPU-MLIR 版本**，旧版编译的 INT8 在 SE9 上会慢于 FP16；另外 `--sq` 会原地平滑权重，同一目录重跑标定前先删除 `.mlir` 与权重文件（脚本内置 md5 防呆，会直接报错拦截）。

**8. 精度不达标怎么办？**

先对齐前处理：`Resize(短边 256, BICUBIC) → CenterCrop(224) → (x/255 - mean)/std`（ImageNet mean/std）。bmcv 与 C++ 走 VPP，不支持 BICUBIC，用 LINEAR 替代，Top-1 与官方口径差 0.3~0.7 点（与 opencv 路径差约 0.2 点），属正常。再确认测的是完整 1000 张子集（每类 1 张），样本更少时波动更大。INT8 比 FP16 低 0.8~1.9 点是该配方（48 个线性层 INT8 + 其余 F16）的正常代价；若明显更差，检查标定集是否按官方 eval 前处理生成、标定目录是否被重复 `--sq` 污染（见上一条）。FP32/FP16 仍低于 83.9 时，核对特征拼接口径（`CLS` + `mean(patch)`，排除 4 个 register token）与线性头是否对应 `dinov2_vitb14_reg4_linear_head`。

> 更多环境安装、通用工具链相关的常见问题，请参考仓库 [FAQ](../../docs/FAQ.md)。
