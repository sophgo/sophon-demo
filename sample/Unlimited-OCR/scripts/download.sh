#!/bin/bash
set -e

res=$(which unzip 2>/dev/null)
if [ $? != 0 ]; then
    echo "Please install unzip on your system!"
    exit 1
fi

echo "Installing/upgrading dfss..."
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade 2>&1 | tail -1

scripts_dir=$(dirname $(readlink -f "$0"))
pushd "$scripts_dir/.."

# ============================================================ models
mkdir -p ./models
if [ ! -f "./models/unlimited_ocr_w4bf16_vit.zip" ] && ! ls ./models/*.bmodel >/dev/null 2>&1; then
    echo "Downloading Unlimited-OCR W4BF16 ViT+LLM bmodel (~2.5GB)..."
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Unlimited-OCR/unlimited_ocr_w4bf16_vit.zip
    unzip -o unlimited_ocr_w4bf16_vit.zip -d ./models/
    rm -f unlimited_ocr_w4bf16_vit.zip
    mkdir -p ./models/config
    for f in embedding.bin tokenizer.json tokenizer_config.json special_tokens_map.json processor_config.json config.json vit_extras.npz; do
        if [ -f "./models/$f" ]; then
            mv "./models/$f" ./models/config/
        fi
    done
    echo "Unlimited-OCR bmodel download and extraction complete!"
else
    echo "Models already exist. Remove ./models/ if you need to re-download."
fi

# ============================================================ datasets
mkdir -p ./datasets
if ! ls ./datasets/*.png >/dev/null 2>&1; then
    echo "Downloading test images (~143KB)..."
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Unlimited-OCR/unlimited_ocr_datasets.zip
    unzip -o unlimited_ocr_datasets.zip -d ./datasets/
    rm -f unlimited_ocr_datasets.zip
    echo "Test images download and extraction complete!"
else
    echo "Datasets already exist. Remove ./datasets/*.png if you need to re-download."
fi

# ============================================================ tpu_mlir_uocr whl
if ! pip3 show tpu_mlir_uocr >/dev/null 2>&1; then
    echo "Downloading tpu_mlir_uocr wheel (~203KB) for model compilation..."
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Unlimited-OCR/tpu_mlir_uocr-1.28.1+uocr-py3-none-any.whl
    echo "To install: pip install tpu_mlir_uocr-1.28.1+uocr-py3-none-any.whl"
else
    echo "tpu_mlir_uocr already installed. Skip download."
fi

popd
echo "All done!"
