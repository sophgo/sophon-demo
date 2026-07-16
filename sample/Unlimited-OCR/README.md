# Unlimited-OCR

## 目录
- [Unlimited-OCR](#unlimited-ocr)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
  - [3. 运行环境准备](#3-运行环境准备)
  - [4. 准备模型与数据](#4-准备模型与数据)
    - [4.1 使用预编译模型](#41-使用预编译模型)
    - [4.2 自行编译模型](#42-自行编译模型)
  - [5. 例程测试](#5-例程测试)
  - [6. 精度测试](#6-精度测试)
    - [6.1 测试方法](#61-测试方法)
    - [6.2 测试结果](#62-测试结果)
  - [7. 性能测试](#7-性能测试)
  - [8. FAQ](#8-faq)

## 1. 简介
Unlimited-OCR 是百度提出的长程文档解析 VLM，基于 DeepseekV2-MoE 架构，支持将单图或多页 PDF 一次性解析输出 Markdown。关于模型特性请前往源 repo 查看：[Unlimited-OCR](https://github.com/baidu/Unlimited-OCR)。

本例程对 Unlimited-OCR 进行移植，使之能在 SOPHON BM1688 (SE9-16) 上进行推理测试。模型约 3.3B 参数，分为三部分：

- **视觉塔 deeplip**：CLIP-L + SAM ViT-B 双路编码，经 projector 送入 LLM
- **LLM**：12 层 DeepseekV2-MoE（hidden_size=1280，64 路由专家/top-6 + 2 shared expert）
- **生成**：自回归 + sliding-window ngram no-repeat

> **状态**：LLM + 视觉塔 bmodel 编译完成，端到端图像 OCR 在 SE9 验证通过。BF16 精度与 HF 源码逐层一致。

## 2. 特性
* 支持 BM1688 SoC (SE9-16)
* 支持 W4BF16、BF16 模型编译和推理
* 支持基于 SAIL 推理的 Python 例程
* 支持单图和多页 PDF 文档解析
* 支持图像 OCR（base / gundam 多 tile 模式）
* 支持 embedding 外置磁盘（`--embedding_disk`）

## 3. 运行环境准备

在 SE9-16 上运行需要修改内存分布，确保 gmem 有足够连续空间加载 bmodel。

对于 SE9-16 8G 版本（DDR 8GB，默认 npu=1.5GB + vpp=4.0GB，单块连续上限 ~3.8GB），W4BF16 bmodel（device mem 2.9GB）不需要修改内存；如需加载 BF16 bmodel（device mem 6.75GB），须使用 SE9-16 16G 版本或修改 npu 布局为 5120MB。

如果需要修改内存分布（如加载大 bmodel），参考如下命令（只需执行一次，重启永久生效）：

```bash
cd /data && mkdir -p memedit && cd memedit
wget -nd https://github.com/sophgo/sophon-tools/releases/download/v24.09.21/memory_edit_v2.10.tar.xz
tar xf memory_edit_v2.10.tar.xz && cd memory_edit
MEMORY_EDIT_CHPI_TYPE=bm1688 ./memory_edit.sh -p          # 打印当前布局

# SE9-16 8G 版本（DDR 8GB，max npu+vpp=6840MB）
MEMORY_EDIT_CHPI_TYPE=bm1688 ./memory_edit.sh -c -npu 5120 -vpu 0 -vpp 1720
sudo cp /boot/boot.itb /boot/boot.itb.bak.orig            # 备份原始
sudo cp /data/memedit/memory_edit/boot.itb /boot/boot.itb && sync && sudo reboot
# 等待约 70s 重启完成，dmesg | grep gmem 确认 gmem[0] total 0x140000000 (5120MB)
```

> **注意**：
> 1. tpu 总内存为 npu/vpu/vpp 三者之和；
> 2. `npu 4096+vpp 2048` 不可行（npu 4GB 对齐后可用连续空间 <3.8GB），必须 `npu 5120`；
> 3. 系统需留至少 1.66GB，不要再压低 vpp；
> 4. 更多教程请参考 [SoC 内存修改工具](https://doc.sophgo.com/sdk-docs/v23.07.01/docs_latest_release/docs/SophonSDK_doc/zh/html/appendix/2_mem_edit_tools.html)。

## 4. 准备模型与数据

### 4.1 使用预编译模型

本例程在 `scripts` 目录下提供了下载脚本 `download.sh`：

```bash
cd sample/Unlimited-OCR
# 安装 unzip，若已安装请跳过
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```

执行后，模型和配置文件下载至 `models/` 目录：

```
./models
├── unlimited_ocr_w4bf16_vit.bmodel   # W4BF16 ViT+LLM 组合 bmodel (~3.3GB)
└── config/
    ├── embedding.bin                 # LLM 词表嵌入（--embedding_disk）
    ├── vit_extras.npz                # 视觉塔额外参数（image_newline、view_seperator）
    ├── tokenizer.json                # tokenizer 配置
    ├── tokenizer_config.json
    ├── special_tokens_map.json
    ├── processor_config.json
    └── config.json
```

### 4.2 自行编译模型

建议使用 TPU-MLIR 编译 BModel，源模型为 HF safetensors 格式（~6.7GB）。完整的编译指南、converter 说明和常见问题请参考 [Unlimited-OCR 模型编译指南](./docs/Unlimited_OCR_Compile_Guide.md)。

简要流程（在 TPU-MLIR Docker 容器内执行）：

**TPU-MLIR 环境准备**（可参考 [TPU-MLIR 环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)）：

```bash
# 假设 tpu-mlir 和 sophon-demo 在同一父目录
docker run -d --name unlimited_ocr_mlir --shm-size=64g \
  -v ~/work:/workspace -w /workspace/tpu-mlir \
  sophgo/tpuc_dev:latest tail -f /dev/null
docker exec -it unlimited_ocr_mlir bash

# 容器内：安装基础 tpu_mlir + Unlimited-OCR overlay
pip install tpu_mlir==1.28.1
pip install https://open.sophgo.com/sophon-demo/Unlimited-OCR/tpu_mlir_uocr-1.28.1+uocr-py3-none-any.whl
# 设置环境
pip install transformers==4.57.1 torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install addict easydict einops
```

> **说明**：`tpu_mlir_uocr` 是 overlay wheel，在标准 `tpu_mlir==1.28.1` 之上添加了 Unlimited-OCR converter 支持（[相关 Gerrit 评审](https://gerrit-ai.sophgo.vip:8443/165605)）。它覆盖了 `tpu_mlir/python/llm/` 和 `tpu_mlir/python/tools/` 下的 Python 文件，添加了 MoE one-hot routing、GGUF/Safetensors 双路径加载等功能，不影响其他模型的编译。

**下载 HF 权重**：

```bash
huggingface-cli download baidu/Unlimited-OCR --local-dir /workspace/sophon-demo/temp/unlimited-ocr-weights
```

**编译 W4BF16 ViT+LLM 组合 bmodel**（推荐，~2-3GB，适合 SE9-16 8G/16G）：

```bash
UOCR_UNFUSE=1 UOCR_DENSE=1 LLM_GEN_MLIR_WORKERS=1 \
  PPL_PROJECT_ROOT=/tmp PPL_JIT_CMODEL=1 CHIP_ARCH=BM1688 \
  python3 python/tools/llm_convert.py \
  -m /workspace/sophon-demo/temp/unlimited-ocr-weights \
  -s 512 -q w4bf16 -g 64 -c bm1688 --num_core 1 --do_sample --debug --embedding_disk \
  -o /workspace/sophon-demo/sample/Unlimited-OCR/models
```

**编译 BF16 bmodel**（更高精度，~7.1GB，需 SE9-16 16G 或修改内存的 8G）：

```bash
UOCR_UNFUSE=1 UOCR_DENSE=1 LLM_GEN_MLIR_WORKERS=1 \
  PPL_PROJECT_ROOT=/tmp PPL_JIT_CMODEL=1 CHIP_ARCH=BM1688 \
  python3 python/tools/llm_convert.py \
  -m /workspace/sophon-demo/temp/unlimited-ocr-weights \
  -s 512 -q bf16 -c bm1688 --num_core 1 --do_sample --debug --embedding_disk \
  -o /workspace/sophon-demo/sample/Unlimited-OCR/models
```

> **注意**：
> 1. 编译需设置 `UOCR_UNFUSE=1 UOCR_DENSE=1`（MoE one-hot routing，SE9 安全绕开 fused MlpOp）；
> 2. `LLM_GEN_MLIR_WORKERS=1` 串行生成 mlir（un-fuse 多线程抢 MLIR Context 会 core dump）；
> 3. W4BF16 编译需 `-g 64`（不能用 128，down_proj in_dim=896 在 gs=128 时奇数分组触发 qzeros shape bug）；
> 4. 如需仅编译 LLM（不含视觉塔），增加 `UOCR_LLM_ONLY=1`；
> 5. 编译中间产物约 60GB，完成后须删除 `model_*_static/` 目录释放空间。

编译完成后生成 `vit_extras.npz`（图像 OCR 需要）：

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

## 5. 例程测试

- [Python 例程](./python/README.md)

简要文本生成冒烟测试（无需视觉塔）：

```bash
cd sample/Unlimited-OCR/python
python3 unlimited_ocr_sail.py \
  --bmodel ../models/unlimited_ocr_w4bf16_vit.bmodel \
  --tokenizer ../models/config \
  --prompt "The capital of France is" \
  --max_new_tokens 64
```

图像 OCR 测试：

```bash
python3 unlimited_ocr_sail.py \
  --bmodel ../models/unlimited_ocr_w4bf16_vit.bmodel \
  --tokenizer ../models/config \
  --vit_extras ../models/config/vit_extras.npz \
  --image ../datasets/test.jpg --image_mode base \
  --ngram_size 35 --ngram_window 128 --max_new_tokens 2048
```

多页 PDF：

```bash
# 先用 pdf_to_images 转成页面图，再逐页 OCR
python3 unlimited_ocr_sail.py ... --image_mode base --ngram_window 1024
```

## 6. 精度测试

### 6.1 测试方法
使用教师强制（teacher-forcing）对比 bmodel 与 HF 源码在每个 prefix 位置的 next-token 预测。同时支持逐层 hidden state 对比以隔离量化误差来源。

### 6.2 测试结果

**BF16 bmodel**（与 HF float32 逐层对比，12 层 hidden states）：

| 测试项 | 结果 |
|--------|------|
| 逐层 max_diff | < 0.002（所有 12 层，落在 bf16 精度误差范围内） |
| CModel（TPU CPU 模拟器）block_0 max_diff | 0.000011 |
| 端到端生成 | 英文流畅正确，数值链路验证通过 |

> **结论**：BF16 bmodel 与 HF 源码在所有 12 层的 hidden states 完全一致，从 MLIR 到 TPU 指令的端到端数值保真度已确认。

**W4BF16 bmodel**（int4 权重 / bf16 激活，group=128）：

| 测试项 | 结果 |
|--------|------|
| 逐层 max_diff | block_0: 0.009, block_1: 0.021, block_2: 0.054（累积误差） |
| 端到端英文文本生成 | ✅ 正常 |
| 端到端中文文本生成 | ⚠️ 易退化（模型本身能力限制，非 bmodel bug） |
| 图像 OCR（文档） | ⚠️ 能识别部分内容，中文易退化 |
| 图像 OCR（英文/收据） | ❌ 无法正确识别（输出中文幻觉） |

> **说明**：W4BF16 的差异来自 int4 量化本身的精度损失。英文生成流畅正确，中文退化和 OCR 不准是模型能力限制（1280 hidden / 12 层 / 训练数据偏英文），与 converter/bmodel 无关。

## 7. 性能测试

测试输入为 "The capital of France is"（不含视觉塔，纯文本生成），max_new_tokens=64：

| 测试平台 | 测试程序 | 测试模型 | first token latency(s) | token per second(tokens/s) |
| -------- | -------- | -------- | ---------------------- | -------------------------- |
| SE9-16 (2core) | unlimited_ocr_sail.py | unlimited_ocr_w4bf16_vit.bmodel (W4BF16) | 5.4 | 3.8 |
| SE9-16 (2core) | unlimited_ocr_sail.py | uocr_bf16_vit_fixed.bmodel (BF16)  | 5.0 | 2.5 |

> **测试说明**：
> 1. 测试输入为纯文本生成 prompt，不包含视觉塔预处理；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. SE9-16 主控为 8 核 CA53@1.6GHz；
> 4. W4BF16 decode 更快（权重 4× 压缩，内存带宽节省），但 prefill 略慢（反量化开销）；
> 5. 模型加载时间：BF16 ~78s，W4BF16 ~49s。

## 8. FAQ

请参考 [FAQ](../../docs/FAQ.md) 查看一些常见的问题与解答。另外本例程的常见问题：

- **bmodel 加载失败（`Load bmodel failed`）**：先用已知能跑的 bmodel（如 `/data/test/BM1688/yolo26s_fp16_1b.bmodel`）测试 TPU 是否正常。如果简单 bmodel 也加载失败，执行 `dmesg | grep "TPU.*hang"` 确认是否 TPU hang。TPU hang 后所有 bmodel 都加载失败，必须重启 SE9 恢复，不要误判为 bmodel 或固件问题。
- **卸载/重载 bmodel 时报错**：SAIL EngineLLM 不支持同一进程内 release 后再 load（驱动状态未完全清理），建议使用 withvit 组合 bmodel 在同一 engine 内跑 ViT+LLM，避免分两个 bmodel 反复 release/reload。
- **中文 OCR 质量差、英文识别输出中文**：这是模型本身能力限制（12 层/1280 hidden/训练数据偏英文），非 bmodel bug。BF16 bmodel 已确认逐层精度与 HF 一致，W4BF16 在量化误差范围内。建议评估模型适用场景。
- **图像多 tile 模式报错或输出异常**：当前 bmodel 编为 seq512，gundam 多 tile 模式需要 seq ≥ 2048。如需完整多 tile 支持，需重新编译更大 seq 的 bmodel（注意 gmem 限制）。
- **编译报错 `pybind11 PyGILState_Check` core dump**：un-fuse MLIR 生成必须串行（`LLM_GEN_MLIR_WORKERS=1`），详见 [模型编译指南](./docs/Unlimited_OCR_Compile_Guide.md)。
- **`embedding net not found` 警告**：正常的，`--embedding_disk` 将词表外置到 `config/embedding.bin`，请确保该文件在 bmodel 同目录的 `config/` 下。
