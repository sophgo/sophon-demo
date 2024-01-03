#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))
outdir=../../../models/processors/bmodels

function gen_mlir()
{
    model_transform.py \
        --model_name openpose_body
        --model_def ./openpose_body_processor.pt \
        --input_shapes [[1,3,184,184]] \
        --mlir openpose_body.mlir

    model_transform.py \
    --model_name openpose_hand
    --model_def ./openpose_hand_processor.pt \
    --input_shapes [[1,3,184,184]] \
    --mlir openpose_hand.mlir
}

function gen_fp16bmodel()
{
    model_deploy.py \
        --mlir openpose_body.mlir \
        --quantize F16 \
        --chip bm1684x \
        --model openpose_body_fp16.bmodel

    mv openpose_body_fp16.bmodel $outdir/

    model_deploy.py \
    --mlir openpose_hand.mlir \
    --quantize F16 \
    --chip bm1684x \
    --model openpose_hand_fp16.bmodel

    mv openpose_hand_fp16.bmodel $outdir/
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi

gen_mlir
gen_fp16bmodel

popd