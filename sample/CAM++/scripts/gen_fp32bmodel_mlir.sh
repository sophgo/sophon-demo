#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))

if [ ! $1 ]; then
    target=bm1684x
    target_dir=BM1684X
else
    target=${1,,}
    target_dir=${target^^}
fi

name=campplus

outdir=../models/$target_dir

function gen_mlir()
{
    model_transform.py \
        --model_name $name \
        --model_def ../models/onnx/${name}.onnx \
        --input_shapes [[$1,600,80]] \
        --dynamic \
        --mlir ${name}_$1b.mlir
}

function gen_fp32bmodel()
{
    model_deploy.py \
        --mlir ${name}_$1b.mlir \
        --quantize F32 \
        --chip $target \
        --dynamic \
        --disable_layer_group \
        --model ${name}_${target}_fp32_$1b.bmodel
    mv ${name}_${target}_fp32_$1b.bmodel $outdir
}

pushd $model_dir
if [ ! -d "$outdir" ]; then
    echo $pwd
    mkdir $outdir
fi

# batch_size=1
gen_mlir 1
gen_fp32bmodel 1

# batch_size=4
#gen_mlir 4
#gen_fp32bmodel 4
popd
