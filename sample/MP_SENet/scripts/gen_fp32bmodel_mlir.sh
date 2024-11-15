#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))

if [ ! $1 ]; then
    target=bm1684x
    target_dir=BM1684X
else
    target=${1,,}
    target_dir=${target^^}
fi

outdir=../models/$target_dir

function gen_mlir()
{
    model_transform.py \
        --model_name mpsenet \
        --model_def ../models/onnx/mpsenet.onnx \
        --input_shapes [[1,201,640],[1,201,640]] \
        --mlir mpsenet_$1b.mlir
}

function gen_fp32bmodel()
{
    model_deploy.py \
        --mlir mpsenet_$1b.mlir \
        --quantize F32 \
        --chip $target \
        --model mpsenet_$1b_fp32.bmodel

    mv mpsenet_$1b_fp32.bmodel $outdir/
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
gen_mlir 1
gen_fp32bmodel 1

popd