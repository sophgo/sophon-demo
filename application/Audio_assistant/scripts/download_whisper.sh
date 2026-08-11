#!/bin/bash
# Download the sail-based combined Whisper bmodel + positional-embedding assets.
# This replaces the old libuntpu 5-bmodel set; see python/bmwhisper/ (copied
# from sample/Whisper, sail-only, no libuntpu dependency).
#
# Usage:
#   ./download_whisper.sh                 # default: model=small, chip=1688
#   ./download_whisper.sh --model small --chip 1688
#   ./download_whisper.sh --chip 1684x    # 1684X PCIe/SoC
set -e

res=$(which unzip)
if [ $? != 0 ]; then
    echo "Please install unzip on your system!"
    exit 1
fi
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade --quiet

scripts_dir=$(dirname $(readlink -f "$0"))
pushd "$scripts_dir"

model="small"
chip="1688"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) model="$2"; shift 2 ;;
        --chip)  chip="$2";  shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# map chip -> (dfss subdir, arch tag in filename)
case "$chip" in
    1688)
        dfss_subdir="BM1688"; arch_tag="1688" ;;
    1684x|1684X)
        dfss_subdir="BM1684X"; arch_tag="1684x" ;;
    *)
        echo "Unsupported chip: $chip (expect 1688 or 1684x)"; exit 1 ;;
esac

bmodel_name="bmwhisper_${model}_${arch_tag}_f16.bmodel"
dfss_url="open@sophgo.com:sophon-demo/Whisper/models/${dfss_subdir}/${bmodel_name}"

# combined bmodel -> ../models/whisper/
mkdir -p ../models/whisper
if [ ! -f "../models/whisper/${bmodel_name}" ]; then
    python3 -m dfss --url="${dfss_url}"
    mv "${bmodel_name}" ../models/whisper/
    echo "bmodel download -> ../models/whisper/${bmodel_name}"
else
    echo "bmodel exists, skip. (remove ../models/whisper/${bmodel_name} to re-download)"
fi

# assets (positional_embedding_*.npz, mel_filters.npz, *.tiktoken)
# -> ../python/bmwhisper/assets/   (chip-independent)
if [ ! -d "../python/bmwhisper/assets" ]; then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Whisper/model_240408/assets.zip
    unzip -o assets.zip -d ../python/bmwhisper
    rm assets.zip
    echo "assets download -> ../python/bmwhisper/assets/"
else
    echo "assets exist, skip. (remove ../python/bmwhisper/assets to re-download)"
fi

popd
echo "done."
