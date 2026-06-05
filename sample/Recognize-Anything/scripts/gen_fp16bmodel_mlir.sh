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
    onnx_path=../models/onnx/ram.onnx
    model_transform.py \
        --model_name $model_name \
        --model_def $onnx_path \
        --input_shapes [[$1,3,384,384]] \
        --pixel_format rgb  \
        --mlir ${model_name}_$1b.mlir
}

function gen_fp16bmodel()
{
    model_deploy.py \
        --mlir ${model_name}_$1b.mlir \
        --quantize F16 \
        --chip $target \
        --model ${model_name}_fp16_$1b.bmodel

    mv ${model_name}_fp16_$1b.bmodel $outdir/
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
model_name=ram
gen_mlir 1
gen_fp16bmodel 1
popd