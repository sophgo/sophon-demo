#!/bin/bash
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

function gen_mlir()
{
    model_transform.py \
        --model_name segformer \
        --keep_aspect_ratio \
        --model_def ../models/onnx/segformer.b0.512x1024.city.160k.onnx \
        --input_shapes [[$1,3,512,1024]] \
        --mean 123.675,116.28,103.53 \
        --mlir segformer.b0.512x1024.city.160k_$1b.mlir 
}

function gen_fp16bmodel()
{
    model_deploy.py \
        --mlir segformer.b0.512x1024.city.160k_$1b.mlir  \
        --quantize F16 \
        --chip $target \
        --model segformer.b0.512x1024.city.160k_fp16_$1b.bmodel

    mv segformer.b0.512x1024.city.160k_fp16_$1b.bmodel $outdir/
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
gen_mlir 1
gen_fp16bmodel 1

popd