#!/bin/bash
# Compile TAPNext++ per-step ONNX graphs to FP16 BModel on BM1688/BM1684X.
#
# TAPNext++ is a recurrent point tracker. We export TWO static-shape ONNX graphs
# (see tools/export_onnx.py):
#   tapnext_init.onnx  : frame + query_points -> tracks, vis, initial state   (2 inputs)
#   tapnext_step.onnx  : frame + state        -> tracks, vis, updated state  (27 inputs)
# The host runs the recurrence loop and feeds the state tensors back each step.
#
# Preprocessing: the ONNX wrappers expect an already-normalized frame in [-1, 1]
# (RGB, NCHW). For FP16, --mean/--scale is calibration-only metadata and NOT a
# runtime op, so we omit it here and do `x/127.5 - 1.0` + BGR->RGB host-side
# (matching the ResNet FP16 convention). All 27 step-graph inputs are treated as
# plain float32 tensors -- the 24 cache tensors + step + query_points are state,
# not images.
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

outdir=../models/$target_dir

# POC static shapes: 256x256, Q=1.
# 1025 = 32*32 video patches (256/8) + 1 query token.
MODEL_SIZE=256
NUM_TOKENS=$(( (MODEL_SIZE/8)*(MODEL_SIZE/8) + 1 ))  # 1025

# Step-graph input shapes: frame, step, query_points, then 12x(rg_lru, conv1d).
step_shapes="[[1,3,${MODEL_SIZE},${MODEL_SIZE}],[1],[1,1,3]"
for i in $(seq 1 12); do
    step_shapes="${step_shapes},[${NUM_TOKENS},768],[${NUM_TOKENS},3,768]"
done
step_shapes="${step_shapes}]"

function gen_mlir()
{
    # $1 = graph name (init|step), $2 = input_shapes
    model_transform.py \
        --model_name tapnext_$1 \
        --model_def ../models/onnx/tapnext_$1.onnx \
        --input_shapes "$2" \
        --mlir tapnext_$1.mlir
}

function gen_fp16bmodel()
{
    # $1 = graph name (init|step)
    model_deploy.py \
        --mlir tapnext_$1.mlir \
        --quantize F16 \
        --chip $target \
        --model tapnext_${1}_fp16_1b.bmodel

    mv tapnext_${1}_fp16_1b.bmodel $outdir/
    if test $target = "bm1688"; then
        model_deploy.py \
            --mlir tapnext_$1.mlir \
            --quantize F16 \
            --chip $target \
            --model tapnext_${1}_fp16_1b_2core.bmodel \
            --num_core 2
        mv tapnext_${1}_fp16_1b_2core.bmodel $outdir/
    fi
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi

# init graph: 2 inputs
gen_mlir init "[[1,3,${MODEL_SIZE},${MODEL_SIZE}],[1,1,3]]"
gen_fp16bmodel init

# step graph: 27 inputs (frame + step + query_points + 24 cache tensors)
gen_mlir step "$step_shapes"
gen_fp16bmodel step

popd
