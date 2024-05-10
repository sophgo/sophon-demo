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

gen_mlir()
{
    model_transform.py \
        --model_name groundingdino \
        --model_def ../models/onnx/GroundingDino.onnx \
        --input_shapes [[1,3,800,800],[1,256],[1,256,256],[1,256],[1,256],[1,256,256],[1,256],[1,13294,4]]  \
        --mlir groundingdino.mlir
}

gen_fp16bmodel()
{
    model_deploy.py \
        --mlir groundingdino.mlir \
        --quantize F16 \
        --chip ${target} \
        --model groundingdino_${target}_fp16.bmodel

    mv groundingdino_${target}_fp16.bmodel $outdir/
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi

# batch size 1
gen_mlir 1
gen_fp16bmodel 1

popd
