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
        --model_name c3d \
        --model_def ../models/onnx/c3d_ucf101.onnx \
        --input_shapes [[$1,3,16,112,112]] \
        --mlir c3d_$1b.mlir
}

function gen_fp32bmodel()
{
    model_deploy.py \
        --mlir c3d_$1b.mlir \
        --quantize F32 \
        --chip $target \
        --model c3d_fp32_$1b.bmodel
    mv c3d_fp32_$1b.bmodel $outdir
    if test $target = "bm1688";then
        model_deploy.py \
            --mlir c3d_$1b.mlir \
            --quantize F32 \
            --chip $target \
            --model c3d_fp32_$1b_2core.bmodel \
            --num_core 2
        mv c3d_fp32_$1b_2core.bmodel $outdir
    fi
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
gen_mlir 4
gen_fp32bmodel 4
popd