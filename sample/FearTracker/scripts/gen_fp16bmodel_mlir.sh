#!/bin/bash
set -e

# ============================================================
# FEARTracker bmodel compilation script
# FEAR is a non-image model with 2 inputs of different sizes.
# Run inside TPU-MLIR Docker container.
# Usage: ./gen_fp16bmodel_mlir.sh [bm1684x|bm1688]
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

MODEL_NAME=feartracker
ONNX_PATH=../models/onnx/feartracker.onnx
outdir=../models/$target_dir

function gen_mlir()
{
    model_transform.py \
        --model_name ${MODEL_NAME} \
        --model_def ${ONNX_PATH} \
        --input_shapes [[1,3,128,128],[1,3,256,256]] \
        --channel_format none \
        --mlir ${MODEL_NAME}.mlir
}

function gen_fp16bmodel()
{
    model_deploy.py \
        --mlir ${MODEL_NAME}.mlir \
        --quantize F16 \
        --chip $target \
        --model ${MODEL_NAME}_fp16_1b.bmodel

    mv ${MODEL_NAME}_fp16_1b.bmodel $outdir/

    if test $target = "bm1688"; then
        model_deploy.py \
            --mlir ${MODEL_NAME}.mlir \
            --quantize F16 \
            --chip $target \
            --model ${MODEL_NAME}_fp16_1b_2core.bmodel \
            --num_core 2
        mv ${MODEL_NAME}_fp16_1b_2core.bmodel $outdir/
    fi
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
echo "  $outdir/${MODEL_NAME}_fp16_1b.bmodel"
if test $target = "bm1688"; then
    echo "  $outdir/${MODEL_NAME}_fp16_1b_2core.bmodel"
fi
echo "=========================================="
popd