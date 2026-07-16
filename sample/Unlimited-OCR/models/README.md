# models/

本目录存放编译产物，**不进 git**（见 `../.gitignore`）。获取方式二选一：

1. **自行编译**（推荐，见 `../README.md` §4）：用 `tpu-mlir` 的 `llm_convert` + 本仓库配套 `UnlimitedOCRConverter.py` 编译。
2. **下载预编译**（待上传）：`bash ../scripts/download.sh`（待 bmodel 上传至 dfss 后启用）。

编译后本目录应包含：
- `unlimited-ocr_w4bf16_seq512_*.bmodel` — combined LLM bmodel（含 12 block + lm_head + greedy/sample_head，可选 in-bmodel `vit` net）
- `config/embedding.bin` — `[129280, 1280]` bf16 embedding table（`--embedding_disk` 编译时由 CPU 查表，bmodel 不含 embedding net）
- `config/tokenizer.json` 等 — tokenizer（从 HF 权重目录拷入）
- `config/vit_extras.npz` — `image_newline` / `view_seperator`（仅图像 OCR 需要，从 HF 权重导出）
