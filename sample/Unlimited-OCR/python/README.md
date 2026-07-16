# Python例程

## 目录
- [Python例程](#python例程)
  - [目录](#目录)
  - [1. 环境准备](#1-环境准备)
    - [1.1 SoC平台](#11-soc平台)
  - [2. 推理测试](#2-推理测试)
    - [2.1 参数说明](#21-参数说明)
    - [2.2 文本生成](#22-文本生成)
    - [2.3 图像OCR](#23-图像ocr)

python目录下提供了一系列Python例程，具体情况如下：

| 文件 | 说明 |
|------|------|
| `unlimited_ocr_sail.py` | 主推理脚本：加载 combined bmodel，prefill + decode 生成循环（KV cache），ngram no-repeat |
| `preprocess.py` | 图像预处理（dynamic_preprocess tiling、归一化、image token 序列构建、prompt 拼接） |
| `ngram_processor.py` | SlidingWindowNoRepeatNgramProcessor（从原模型 1:1 移植） |
| `pdf_to_images.py` | PyMuPDF PDF → 页面图 |
| `requirements.txt` | Python 依赖列表 |

## 1. 环境准备
### 1.1 SoC平台

本例程仅支持 BM1688 SoC（SE9-16）。刷机后在 `/opt/sophon/` 下已预装 libsophon、sophon-opencv 和 sophon-ffmpeg 运行库包，sophon-sail 已随 SDK 预装。

安装其他第三方库：

```bash
pip3 install -r python/requirements.txt
```

主要依赖：`ml_dtypes`、`Pillow`、`PyMuPDF`、`numpy`、`sentencepiece`、`tiktoken`。

## 2. 推理测试
python例程不需要编译，可以直接运行。

### 2.1 参数说明
`unlimited_ocr_sail.py` 主要参数：

```bash
usage: unlimited_ocr_sail.py [--bmodel BMODEL] [--vit_bmodel VIT_BMODEL] [--tokenizer TOKENIZER]
                             [--vit_extras VIT_EXTRAS] [--embedding_bin EMBEDDING_BIN]
                             [--image IMAGE] [--prompt PROMPT] [--image_mode IMAGE_MODE]
                             [--max_new_tokens MAX_NEW_TOKENS] [--ngram_size NGRAM_SIZE]
                             [--ngram_window NGRAM_WINDOW] [--dev DEV]
--bmodel:          用于推理的 combined bmodel 路径（含 ViT+LLM）
--vit_bmodel:      独立视觉塔 bmodel 路径（仅当 combined bmodel 不含 vit net 时需要）
--tokenizer:       tokenizer 配置目录路径（含 tokenizer.json 等）
--vit_extras:      vit_extras.npz 路径（图像 OCR 需要，含 image_newline / view_seperator）
--embedding_bin:   config/embedding.bin 路径（--embedding_disk bmodel 必须，默认自动发现）
--image:           输入图片路径（无图片则为纯文本生成）
--prompt:          提示词，图片模式下需包含 <image> 占位符，默认 "<image>document parsing."
--image_mode:      图像模式，可选 gundam（多 tile 切块）或 base（单 1024×1024），默认 gundam
--max_new_tokens:  最大生成 token 数，默认 2048
--ngram_size:      ngram no-repeat 窗口大小，默认 35
--ngram_window:    ngram 抑制窗口长度，单图 128、多页 PDF 1024
--dev:             TPU 设备 id，默认 0
```

### 2.2 文本生成
冒烟测试，无需视觉塔。用于验证 bmodel 加载和 LLM 推理链路正常：

```bash
cd python
python3 unlimited_ocr_sail.py \
  --bmodel ../models/unlimited_ocr_w4bf16_vit.bmodel \
  --tokenizer ../models/config \
  --prompt "The capital of France is" \
  --max_new_tokens 64
```

批量测试 `datasets/test_prompts.txt` 中的提示词：

```bash
while IFS= read -r line; do
    [[ "$line" =~ ^# ]] || [ -z "$line" ] && continue
    echo "=== $line ==="
    python3 unlimited_ocr_sail.py \
      --bmodel ../models/unlimited_ocr_w4bf16_vit.bmodel \
      --tokenizer ../models/config \
      --prompt "$line" --max_new_tokens 64
done < ../datasets/test_prompts.txt
```

> **说明**：Unlimited-OCR 是视觉语言模型，纯文本生成（无图像）仅用于冒烟测试。英文生成流畅正确，中文易退化（模型能力限制，非 bmodel bug）。

### 2.3 图像OCR

单图 OCR（base 模式，1024×1024 单 tile）：

```bash
python3 unlimited_ocr_sail.py \
  --bmodel ../models/unlimited_ocr_w4bf16_vit.bmodel \
  --tokenizer ../models/config \
  --vit_extras ../models/config/vit_extras.npz \
  --image ../datasets/doc_chinese.png --image_mode base \
  --prompt "<image>document parsing." \
  --ngram_size 35 --ngram_window 128 --max_new_tokens 2048
```

多页 PDF：先用 `pdf_to_images.py` 将 PDF 转为页面图，再逐页调用（`--image_mode base --ngram_window 1024`）：

```bash
python3 -c "
from pdf_to_images import pdf_to_images
images = pdf_to_images('your_document.pdf')
print(f'PDF converted to {len(images)} pages')
# 逐页保存后再用 unlimited_ocr_sail.py 处理
"
```

> **注意**：
> 1. 图像 OCR 需要 bmodel 包含视觉塔 net（编译时不加 `UOCR_LLM_ONLY=1`），且 `config/` 下有 `vit_extras.npz`；
> 2. `gundam` 多 tile 模式需要 seq ≥ 2048，当前预编译 bmodel 为 seq512，部分多 tile 场景可能受限；
> 3. OCR 质量受限于模型能力（1280 hidden / 12 层），英文文档识别优于中文。
