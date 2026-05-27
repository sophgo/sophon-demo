#!/bin/bash
set -e

# ============================================================
# Silero VAD bmodel compilation script for BM1684X
# Run inside TPU-MLIR Docker container
# Usage: ./gen_f16_bmodel.sh [bm1684x]
# ============================================================
model_dir=$(dirname $(readlink -f "$0"))

if [ ! $1 ]; then
    target=bm1684x
    target_dir=BM1684X
else
    target=${1,,}
    target_dir=${target^^}
    if test $target = "bm1684"
    then
        echo "bm1684 do not support fp16"
        exit
    fi
fi

MODEL_NAME=silero_vad
ONNX_PATH=../tools/silero_vad_core_clean.onnx
outdir=../models/$target_dir

function gen_mlir()
{
    # 3 inputs: x=[1,576] h=[1,128] c=[1,128]
    # channel_format=none: audio model, no image preprocessing
    model_transform.py \
        --model_name ${MODEL_NAME} \
        --model_def ${ONNX_PATH} \
        --input_shapes [[1,576],[1,128],[1,128]] \
        --channel_format none \
        --mlir ${MODEL_NAME}.mlir
}

function gen_fp16bmodel()
{
    model_deploy.py \
        --mlir ${MODEL_NAME}.mlir \
        --quantize F16 \
        --chip $target \
        --model ${MODEL_NAME}_${target}_f16.bmodel

    mv ${MODEL_NAME}_${target}_f16.bmodel $outdir/
}

pushd $model_dir
if [ ! -d "$outdir" ]; then
    mkdir -p $outdir
fi

echo "=========================================="
echo "Step 1: model_transform (ONNX → MLIR)"
echo "=========================================="
gen_mlir

echo ""
echo "=========================================="
echo "Step 2: model_deploy (MLIR → F16 bmodel)"
echo "=========================================="
gen_fp16bmodel

echo ""
echo "=========================================="
echo "Done! Model at:"
echo "  $outdir/${MODEL_NAME}_${target}_f16.bmodel"
echo "=========================================="
popd