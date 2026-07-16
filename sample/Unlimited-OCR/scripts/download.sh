#!/bin/bash
# Unlimited-OCR 预编译 W4BF16 bmodel + 配置文件下载脚本
#
# 从 Sophgo dfss 服务器拉取预编译的 W4BF16 ViT+LLM 组合 bmodel
# 和 config/（embedding.bin + tokenizer.json + vit_extras.npz 等）。
#
# 适用：BM1688 / SE9-16（8G 和 16G 版本均可加载，device mem ~2.9GB）

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

# models
if [ ! -d "./models" ]; then
    mkdir -p ./models
fi

if [ ! -f "./models/unlimited_ocr_w4bf16_vit.zip" ] && \
   ! ls ./models/*.bmodel >/dev/null 2>&1; then
    echo "Downloading Unlimited-OCR W4BF16 ViT+LLM bmodel (~2.5GB)..."
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Unlimited-OCR/unlimited_ocr_w4bf16_vit.zip
    unzip -o unlimited_ocr_w4bf16_vit.zip -d ./models/
    rm -f unlimited_ocr_w4bf16_vit.zip
    # Move config files to config/ subdirectory
    mkdir -p ./models/config
    for f in embedding.bin tokenizer.json tokenizer_config.json special_tokens_map.json config.json processor_config.json vit_extras.npz; do
        if [ -f "./models/$f" ]; then
            mv "./models/$f" ./models/config/
        fi
    done
    echo "Unlimited-OCR bmodel download and extraction complete!"
else
    echo "Models already exist. Remove ./models/ if you need to re-download."
fi

popd
