# Unlimited-OCR 模型编译指南

## 目录
- [1. 模型架构概述](#1-模型架构概述)
- [2. Converter 说明](#2-converter-说明)
- [3. 环境准备](#3-环境准备)
- [4. 编译流程](#4-编译流程)
  - [4.1 W4BF16 编译（推荐）](#41-w4bf16-编译推荐)
  - [4.2 BF16 编译](#42-bf16-编译)
  - [4.3 备选：GGUF 编译](#43-备选gguf-编译)
  - [4.4 仅编译 LLM（不含视觉塔）](#44-仅编译-llm不含视觉塔)
- [5. 产物说明](#5-产物说明)
- [6. 编译参数说明](#6-编译参数说明)
- [7. 常见问题](#7-常见问题)
- [8. 精度验证记录](#8-精度验证记录)

## 1. 模型架构概述

Unlimited-OCR 约 3.3B 参数 DeepseekV2-MoE VLM，三部分：

- **视觉塔 deeplip**（`gen_vit_mlir`）：CLIP-L（24 层 1024 维，`qkv_proj` 融合）+ SAM ViT-B（12 块 768 维，`qkv` 融合，`neck`/`net_2`/`net_3` 卷积下采样到 1024ch×16×16）。两路 concat 成 2048 维，经 `projector`（Linear 2048→1280）送入 LLM。
- **LLM**（`_gen_unfuse_block_mlir`）：12 层 DeepseekV2。layer 0 稠密 MLP（intermediate 6848），layer 1-11 MoE（64 路由专家 / top-6，`moe_intermediate_size`=896，2 shared expert 合并为单 MLP intermediate=1792）。`hidden_size`=1280，10 heads MHA，`head_dim`=128。
  - Attention：标准 Llama RoPE（`rope_theta`=10000，非 MLA），`sliding_window=128` 仅影响 KV cache（v1 降级为全注意力）。
  - MoE 门控：softmax + greedy top-k，`norm_topk_prob=False`，shared expert 无 sigmoid gate。
- **生成**：自回归 + sliding-window ngram no-repeat。

## 2. Converter 说明

Converter 位于 tpu-mlir 源码树 `python/llm/UnlimitedOCRConverter.py`（继承 `LlmConverter`，使用 QWEN2_MOE 路径）。

### 关键设计决策

#### MoE unfuse（one-hot routing）
SE9-16 的 fused `MlpOp`（MultiExpert）会触发 TPU hang（`bm-sophon0: TPU SYS hang`）。因此使用 `UOCR_UNFUSE=1` 将所有 `moe_fused_mlp_1/2/3` 展开为 Stock MLP + one-hot 路由：
- `CompareOp(Equal)` 比较 TopK indices 与 `expert_range` 选择 expert
- 每个 expert 编译为独立 Stock MLP（192 op/block）

#### expert_range dtype 修复（2026-07-15）
初版 converter 将 `expert_range = [0,1,...,63]` 存为 `np.int32`。MLIR 的 Weight op 默认以 F32 加载，int32 bit pattern 被当作 float32 解释：int32 `0x00000001` → float32 `~1.4e-45`（subnormal），CompareOp 永远匹配不上，routing 全零（所有 MoE expert 贡献为 0）。

**修复**（一行）：`dtype=np.int32` → `dtype=np.float32`

**教训**：任何送入 MLIR Weight op 的非浮点数据必须显式转换为 `np.float32`。

## 3. 环境准备

在 tpu-mlir Docker 容器中编译（可参考 [TPU-MLIR 环境搭建](../../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)）：

```bash
# 假设目录结构：~/work/tpu-mlir/ 和 ~/work/sophon-demo/
docker run -d --name unlimited_ocr_mlir --shm-size=64g \
  -v ~/work:/workspace -w /workspace/tpu-mlir \
  sophgo/tpuc_dev:latest tail -f /dev/null
docker exec -it unlimited_ocr_mlir bash

# 容器内：
source /workspace/tpu-mlir/envsetup.sh       # 设置 PYTHONPATH / LD_LIBRARY_PATH / TPUC_ROOT
pip install transformers==4.57.1 torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install addict easydict einops
pip install "huggingface_hub[cli]"           # hf download 命令需要
```

下载权重：

```bash
huggingface-cli download baidu/Unlimited-OCR --local-dir /workspace/sophon-demo/temp/unlimited-ocr-weights
# ~6.7GB safetensors + modeling 代码 + tokenizer
```

> **磁盘空间**：编译中间产物约 60GB，请确保 ≥70GB 剩余空间。编译完成后必须删除中间产物（`rm -rf model_*_static/`）。

## 4. 编译流程

### 4.1 W4BF16 编译（推荐）

从 HF safetensors 直接 RTN int4 量化，跳过 GGUF 双重量化。产物 ~2-3GB，适合 SE9-16 8G/16G。

```bash
source envsetup.sh
UOCR_UNFUSE=1 UOCR_DENSE=1 LLM_GEN_MLIR_WORKERS=1 \
  PPL_PROJECT_ROOT=/tmp PPL_JIT_CMODEL=1 CHIP_ARCH=BM1688 \
  python3 python/tools/llm_convert.py \
  -m /workspace/sophon-demo/temp/unlimited-ocr-weights \
  -s 512 -q w4bf16 -g 64 -c bm1688 --num_core 1 --do_sample --debug --embedding_disk \
  -o /workspace/sophon-demo/sample/Unlimited-OCR/models
```

### 4.2 BF16 编译

更高精度，bmodel 更大（~7.1GB），需 SE9-16 16G 版本：

```bash
source envsetup.sh
UOCR_UNFUSE=1 UOCR_DENSE=1 LLM_GEN_MLIR_WORKERS=1 \
  PPL_PROJECT_ROOT=/tmp PPL_JIT_CMODEL=1 CHIP_ARCH=BM1688 \
  python3 python/tools/llm_convert.py \
  -m /workspace/sophon-demo/temp/unlimited-ocr-weights \
  -s 512 -q bf16 -c bm1688 --num_core 1 --do_sample --debug --embedding_disk \
  -o /workspace/sophon-demo/sample/Unlimited-OCR/models
```

### 4.3 备选：GGUF 编译

当只有 GGUF 量化权重（`Unlimited-OCR-Q4_K_M.gguf`，~1.9GB）时使用。加 `UOCR_GGUF_RTN=1` 会 dequant GGUF→float→RTN int4。

> **注意**：GGUF Q4_K_M 是双重量化，精度差（教师强制 0/13 匹配），不推荐。仅当无法获取 HF safetensors 时使用。

```bash
source envsetup.sh
UOCR_GGUF_RTN=1 UOCR_UNFUSE=1 UOCR_DENSE=1 LLM_GEN_MLIR_WORKERS=1 \
  PPL_PROJECT_ROOT=/tmp PPL_JIT_CMODEL=1 CHIP_ARCH=BM1688 \
  python3 python/tools/llm_convert.py \
  -m /workspace/sophon-demo/temp/unlimited-ocr-gguf/Unlimited-OCR-Q4_K_M.gguf \
  -s 512 -q w4bf16 -g 64 -c bm1688 --num_core 1 --do_sample --debug --embedding_disk \
  -o /workspace/sophon-demo/sample/Unlimited-OCR/models
```

### 4.4 仅编译 LLM（不含视觉塔）

加 `UOCR_LLM_ONLY=1`，bmodel 不含 `vit` net，仅供文本生成冒烟测试：

```bash
UOCR_UNFUSE=1 UOCR_DENSE=1 UOCR_LLM_ONLY=1 LLM_GEN_MLIR_WORKERS=1 \
  PPL_PROJECT_ROOT=/tmp PPL_JIT_CMODEL=1 CHIP_ARCH=BM1688 \
  python3 python/tools/llm_convert.py \
  -m /workspace/sophon-demo/temp/unlimited-ocr-weights \
  -s 512 -q w4bf16 -g 64 -c bm1688 --num_core 1 --do_sample --debug --embedding_disk \
  -o /workspace/sophon-demo/sample/Unlimited-OCR/models
```

## 5. 产物说明

| 文件 | 说明 |
|------|------|
| `unlimited_ocr_w4bf16_vit.bmodel` | W4BF16 ViT+LLM 组合 bmodel ~3.3GB |
| `config/embedding.bin` | LLM 词表嵌入（316MB，`--embedding_disk` 外置） |
| `config/tokenizer.json` 等 | tokenizer 配置（`llm_convert` 自动从 HF 目录拷贝） |
| `config/vit_extras.npz` | 视觉塔参数（需手动从 HF 权重导出） |

W4BF16 combined bmodel 含 27 net：lm_head + greedy/sample_head + 12 block_cache + 12 block（无 embedding net）+ vit。

导出 `vit_extras.npz`：

```bash
python3 -c "
from safetensors.torch import load_file
import numpy as np
m = load_file('/workspace/sophon-demo/temp/unlimited-ocr-weights/model-00001-of-000001.safetensors')
np.savez('/workspace/sophon-demo/sample/Unlimited-OCR/models/config/vit_extras.npz',
    **{k: m[k].numpy() for k in ['model.image_newline', 'model.view_seperator']})
print('saved vit_extras.npz')
"
```

> 如果 `config/` 目录下缺少 tokenizer 文件（如 GGUF 编译路径，GGUF 不含 tokenizer），需从 HF 权重目录手动拷贝：
> ```bash
> cp /workspace/sophon-demo/temp/unlimited-ocr-weights/{tokenizer.json,tokenizer_config.json,special_tokens_map.json,processor_config.json,config.json} models/config/
> ```

## 6. 编译参数说明

| 环境变量 / 参数 | 说明 |
|----------------|------|
| `UOCR_UNFUSE=1` | MoE one-hot routing（必须，否则 fused MlpOp 在 SE9 触发 TPU hang） |
| `UOCR_DENSE=1` | layer 0 为 Dense MLP（非 MoE）+ layer 1-11 为 MoE |
| `UOCR_LLM_ONLY=1` | 仅编译 LLM，不含视觉塔 |
| `UOCR_GGUF_RTN=1` | GGUF→float→RTN int4（备选路径，不推荐） |
| `LLM_GEN_MLIR_WORKERS=1` | 串行生成 MLIR（必须，un-fuse 多线程抢 MLIR Context 会 core dump） |
| `PPL_PROJECT_ROOT=/tmp` | 修复 tpuc-opt AutoTuner segfault |
| `-q w4bf16 -g 64` | W4BF16 量化，group=64（必须 64，128 会触发 down_proj qzeros shape bug） |
| `-s 512` | 序列长度（受 gmem 连续大小限制） |
| `--embedding_disk` | embedding 外置 CPU（bmodel 不含 embedding net） |
| `--do_sample` | 暴露 logits 子网，支持 ngram no-repeat 抑制 |

## 7. 常见问题

### 编译报错 `pybind11 PyGILState_Check` core dump
un-fuse MLIR 生成必须串行（`LLM_GEN_MLIR_WORKERS=1` 或 `--debug`）。un-fuse 的 192 op/block 在多线程下抢 MLIR Context 触发此 bug。

### 编译报错 `tpuc-opt --address-assign` segfault
设 `PPL_PROJECT_ROOT=/tmp PPL_JIT_CMODEL=1 CHIP_ARCH=BM1688`（修复 AutoTuner `getenv(NULL)` 崩溃）。

### W4BF16 `-g 128` 报错 qzeros shape mismatch
down_proj in_dim=896，gs=128 得 7 组（奇数），触发 stock mlp() qzeros shape bug（7//2=3 vs codegen 期望 ceil=4）。必须用 `-g 64`（得 14/20 组，偶数）。

### BF16 sail bf16 I/O
`out[i].asnumpy()` 返回 uint16（bf16 raw bits），转 float 需 `.view(ml_dtypes.bfloat16).astype(np.float32)`，不能直接 `.astype(np.float32)`。

### 编译产物清理
un-fuse full build 中间产物 ~52GB，编译完只保留 combined `.bmodel` + `config/`，其余删除（`rm -rf model_*_static/`）。Docker 容器产生的文件属 root，需通过 `docker exec <容器名> bash -lc 'rm -rf /workspace/sophon-demo/temp/<目录>'` 删除。

### SE9 TPU hang
若 fused MlpOp 或异常把 TPU 跑挂，**所有 bmodel 都加载失败**（含 released yolo26s），极易误判成"bmodel 编错/固件不兼容"。遇 `ioclt ret=-1 314` / `a53lite load library` 错误先 `dmesg | grep "TPU.*hang"` 确认，再重启 SE9，不要去改 bmodel/固件。

## 8. 精度验证记录

### BF16 bmodel vs HF float32（逐层 hidden state 对比，修复 expert_range 后）

| Block | BModel (BF16) | HF (float32) | Max Diff |
|-------|---------------|-------------|----------|
| 0 | [0.1533, -0.0049, 0.0161, -0.0422, -0.1001] | [0.1528, -0.0049, 0.0157, -0.0424, -0.1001] | 0.0006 |
| 1 | [0.1182, -0.1367, 0.0078, 0.0084, -0.0806] | [0.1177, -0.1370, 0.0077, 0.0084, -0.0804] | 0.0005 |
| 2 | [0.1533, -0.1348, -0.0293, 0.0483, -0.0879] | [0.1528, -0.1347, -0.0288, 0.0492, -0.0871] | 0.0010 |
| 3–11 | 全部无 NaN/Inf | — | < 0.002 |

CModel（TPU CPU 模拟器）block_0 max_diff = 0.000011，从 MLIR 到 TPU 指令的端到端数值保真度确认无误。

### W4BF16 bmodel vs HF float32

| Block | Max Diff vs HF |
|-------|---------------|
| 0 | 0.009 |
| 1 | 0.021（旧版 bug 时为 0.060） |
| 2 | 0.054（W4 量化累积误差） |

W4BF16 的差异来自 int4 量化本身的精度损失，非 converter 或 bmodel bug。

### 端到端生成测试

| 测试 | 结果 |
|------|------|
| 英文 text gen："What is the capital of France?" | ✅ "The capital of France is located at the top of the Seine, which is a river." |
| 中文 text gen："你好，请问你叫什么名字？" | ⚠️ 退化（模型本身中文能力弱） |
| 图像 OCR：中文文档（base mode） | ⚠️ 能识别部分内容，易退化 |
| 图像 OCR：英文/收据 | ❌ 输出中文幻觉（模型训练数据偏中文） |

### V1 已知设计限制

- `sliding_window=128`：TPU-MLIR LLM attention 用全 KV cache，无原生滑动窗，v1 全注意力近似。
- SAM 视觉塔 `rel_pos_h/rel_pos_w` 相对位置注意力 + window/global 分区：v1 降级为标准全注意力（精度有损）。
- SAM neck `LayerNorm2d`：v1 用标准 LayerNorm 在通道轴近似。
- 视觉 bmodel 固定 1024×1024 输入，gundam 640×640 切片需 dynamic-shape 变体。
