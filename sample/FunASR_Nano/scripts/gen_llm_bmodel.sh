#!/bin/bash
# ==============================================================================
# FunASR Nano — Qwen3-0.6B LLM 解码器编译 (w4bf16, via llm_convert.py)
#
# 在 sophon-llm 容器内运行:
#   bash gen_llm_bmodel.sh [bm1684x|bm1688|cv84x6] [num_core]
#
# 前置: 先在 host 运行 tools/extract_llm_weights.py 生成的 HF 模型目录
#       tools/qwen3_0.6b_llm/ (含 pytorch_model.bin + config.json + tokenizer)
#
# 产物: qwen3_0.6b_llm_w4bf16_seq512_<chip>_<core>dev_static_*.bmodel + config/
# ==============================================================================
set -e

if [ ! $1 ]; then
    chip=bm1684x
else
    chip=${1,,}
fi
num_core=${2:-1}
scripts_dir=$(dirname $(readlink -f "$0"))

hf_dir=$scripts_dir/../tools/qwen3_0.6b_llm
out_dir=$scripts_dir/../models/${chip^^}

if [ ! -f "$hf_dir/pytorch_model.bin" ]; then
    echo "ERROR: $hf_dir/pytorch_model.bin not found."
    echo "Run first on host: python3 tools/extract_llm_weights.py"
    exit 1
fi

mkdir -p "$out_dir"
echo "=== Compile Qwen3-0.6B LLM w4bf16 ($chip, ${num_core}core) ==="

llm_convert \
    -m "$hf_dir" \
    -c "$chip" \
    --quantize w4bf16 \
    --num_core "$num_core" \
    --max_input_length 256 \
    -s 512 \
    --out_dir "$out_dir/llm_out"

echo ""
echo "=== LLM compilation done ($chip) ==="
ls -lh "$out_dir/llm_out/"*.bmodel 2>/dev/null
echo ""
echo "Next: copy the .bmodel to models/${chip^^}/ and the tokenizer config"
echo "(config/ subdir produced by llm_convert) to python/config/."
