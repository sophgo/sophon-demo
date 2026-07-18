#!/bin/bash
# ==============================================================================
# FunASR Nano-2512 FP32 BModel 编译 (TPU-MLIR)
# 在 TPU-MLIR 容器内运行: bash gen_fp32bmodel_mlir.sh [bm1688|bm1684x]
# 产物: funasr_encoder_f32_1b.bmodel, funasr_adapter_f32_1b.bmodel
#
# 注: BM1688 上推荐使用 F16（体积更小 462MB vs 911MB，精度相似）
#     编译 F16 请运行 gen_fp16bmodel_mlir.sh
# ==============================================================================
model_dir=$(dirname $(readlink -f "$0"))

if [ ! $1 ]; then
    target=bm1688
    target_dir=BM1688
else
    target=${1,,}
    target_dir=${target^^}
fi

outdir=../models/$target_dir
onnxdir=../models/onnx

function gen_encoder_mlir()
{
    model_transform.py \
        --model_name funasr_encoder_f32 \
        --model_def $onnxdir/sanm_encoder.onnx \
        --input_shapes [[1,200,560],[1]] \
        --mlir funasr_encoder_f32.mlir
}

function gen_encoder_bmodel()
{
    model_deploy.py \
        --mlir funasr_encoder_f32.mlir \
        --quantize F32 \
        --chip $target \
        --model funasr_encoder_f32_1b.bmodel

    mv funasr_encoder_f32_1b.bmodel $outdir/

    if test $target = "bm1688"; then
        model_deploy.py \
            --mlir funasr_encoder_f32.mlir \
            --quantize F32 \
            --chip $target \
            --model funasr_encoder_f32_1b_2core.bmodel \
            --num_core 2
        mv funasr_encoder_f32_1b_2core.bmodel $outdir/
    fi
}

function gen_adapter_mlir()
{
    model_transform.py \
        --model_name funasr_adapter_f32 \
        --model_def $onnxdir/audio_adapter.onnx \
        --input_shapes [[1,200,512],[1]] \
        --mlir funasr_adapter_f32.mlir
}

function gen_adapter_bmodel()
{
    model_deploy.py \
        --mlir funasr_adapter_f32.mlir \
        --quantize F32 \
        --chip $target \
        --model funasr_adapter_f32_1b.bmodel

    mv funasr_adapter_f32_1b.bmodel $outdir/

    if test $target = "bm1688"; then
        model_deploy.py \
            --mlir funasr_adapter_f32.mlir \
            --quantize F32 \
            --chip $target \
            --model funasr_adapter_f32_1b_2core.bmodel \
            --num_core 2
        mv funasr_adapter_f32_1b_2core.bmodel $outdir/
    fi
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi

echo "=== FunASR Nano FP32 BModel Compilation ($target) ==="

echo "[1/2] SANM Encoder FP32 ($target)..."
gen_encoder_mlir
gen_encoder_bmodel
echo "  -> $outdir/funasr_encoder_f32_1b.bmodel"

echo "[2/2] Audio Adapter FP32 ($target)..."
gen_adapter_mlir
gen_adapter_bmodel
echo "  -> $outdir/funasr_adapter_f32_1b.bmodel"

# Clean intermediates except bmodel
rm -f funasr_encoder_f32.mlir funasr_adapter_f32.mlir
rm -f *_origin.mlir *_tpu.mlir *_final.mlir
rm -f *.npz *.json *.profile

echo ""
echo "=== FP32 compilation complete! ==="
ls -lh "$outdir/"*f32*.bmodel
popd
