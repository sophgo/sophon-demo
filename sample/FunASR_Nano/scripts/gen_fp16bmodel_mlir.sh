#!/bin/bash
# ==============================================================================
# FunASR Nano-2512 F16 BModel 编译 (TPU-MLIR ≥ v1.28.1)
# 在 TPU-MLIR 容器内运行: bash gen_fp16bmodel_mlir.sh [bm1688|bm1684x]
# 产物: funasr_encoder_f16_1b.bmodel, funasr_adapter_f16_1b.bmodel
# ==============================================================================
model_dir=$(dirname $(readlink -f "$0"))

if [ ! $1 ]; then
    target=bm1688
    target_dir=BM1688
else
    target=${1,,}
    target_dir=${target^^}
    if test $target = "bm1684"
    then
        echo "bm1684 do not support fp16"
        exit
    fi
fi

outdir=../models/$target_dir
onnxdir=../models/onnx

# SANM Encoder: grid=200 (≈12s audio)
function gen_encoder_mlir()
{
    model_transform.py \
        --model_name funasr_encoder \
        --model_def $onnxdir/sanm_encoder.onnx \
        --input_shapes [[1,200,560],[1]] \
        --mlir funasr_encoder_f16.mlir
}

function gen_encoder_bmodel()
{
    model_deploy.py \
        --mlir funasr_encoder_f16.mlir \
        --quantize F16 \
        --chip $target \
        --model funasr_encoder_f16_1b.bmodel

    mv funasr_encoder_f16_1b.bmodel $outdir/

    if test $target = "bm1688"; then
        model_deploy.py \
            --mlir funasr_encoder_f16.mlir \
            --quantize F16 \
            --chip $target \
            --model funasr_encoder_f16_1b_2core.bmodel \
            --num_core 2
        mv funasr_encoder_f16_1b_2core.bmodel $outdir/
    fi
}

# Audio Adapter: grid=200
function gen_adapter_mlir()
{
    model_transform.py \
        --model_name funasr_adapter \
        --model_def $onnxdir/audio_adapter.onnx \
        --input_shapes [[1,200,512],[1]] \
        --mlir funasr_adapter_f16.mlir
}

function gen_adapter_bmodel()
{
    model_deploy.py \
        --mlir funasr_adapter_f16.mlir \
        --quantize F16 \
        --chip $target \
        --model funasr_adapter_f16_1b.bmodel

    mv funasr_adapter_f16_1b.bmodel $outdir/

    if test $target = "bm1688"; then
        model_deploy.py \
            --mlir funasr_adapter_f16.mlir \
            --quantize F16 \
            --chip $target \
            --model funasr_adapter_f16_1b_2core.bmodel \
            --num_core 2
        mv funasr_adapter_f16_1b_2core.bmodel $outdir/
    fi
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi

echo "=== FunASR Nano F16 BModel Compilation ($target) ==="

# Encoder
echo "[1/2] SANM Encoder F16 ($target)..."
gen_encoder_mlir
gen_encoder_bmodel
echo "  -> $outdir/funasr_encoder_f16_1b.bmodel"

# Adapter
echo "[2/2] Audio Adapter F16 ($target)..."
gen_adapter_mlir
gen_adapter_bmodel
echo "  -> $outdir/funasr_adapter_f16_1b.bmodel"

# Clean intermediates
rm -f funasr_encoder_f16.mlir funasr_adapter_f16.mlir
rm -f *_origin.mlir *_tpu.mlir *_final.mlir
rm -f *.npz *.json *.profile

echo ""
echo "=== F16 compilation complete! ($target) ==="
ls -lh "$outdir/"*f16*.bmodel 2>/dev/null
popd
