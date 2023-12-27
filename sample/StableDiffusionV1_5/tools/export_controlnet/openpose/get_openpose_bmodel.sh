#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))
outdir=../../../models/controlnets/bmodels

function gen_mlir()
{
    model_transform.py \
        --model_name openpose_controlnet \
        --model_def ./openpose_controlnet.pt \
        --input_shapes [[2,4,64,64],[1],[2,77,768],[2,3,512,512]] \
        --mlir openpose_controlnet.mlir
}

function gen_fp16bmodel()
{
    model_deploy.py \
        --mlir openpose_controlnet.mlir \
        --quantize F16 \
        --chip bm1684x \
        --model openpose_controlnet_fp16.bmodel

    mv openpose_controlnet_fp16.bmodel $outdir/
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi

gen_mlir
gen_fp16bmodel

popd
