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
        --model_name resnet50_$1b \
        --model_def ../models/onnx/resnet50_dynamic.onnx \
        --input_shapes [[$1,3,224,224]] \
        --keep_aspect_ratio \
        --pixel_format rgb  \
        --mlir resnet50_$1b.mlir
}

function gen_fp32bmodel()
{
    model_deploy.py \
        --mlir resnet50_$1b.mlir \
        --quantize F32 \
        --chip $target \
        --model resnet50_fp32_$1b.bmodel

    mv resnet50_fp32_$1b.bmodel $outdir/
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
gen_mlir 1
gen_fp32bmodel 1

popd
