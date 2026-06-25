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
        --model_name lprnet \
        --model_def ../models/onnx/lprnet_$1b.onnx \
        --input_shapes [[$1,3,24,94]] \
        --mean 127.5,127.5,127.5 \
        --scale 0.0078125,0.0078125,0.0078125 \
        --keep_aspect_ratio \
        --pixel_format rgb  \
        --mlir lprnet_$1b.mlir
}

function gen_fp32bmodel()
{
    model_deploy.py \
        --mlir lprnet_$1b.mlir \
        --quantize F32 \
        --chip $target \
        --model lprnet_fp32_$1b.bmodel
    mv lprnet_fp32_$1b.bmodel $outdir/
    
    if test $target = "bm1688";then
        model_deploy.py \
            --mlir lprnet_$1b.mlir \
            --quantize F32 \
            --chip $target \
            --num_core 2 \
            --model lprnet_fp32_$1b_2core.bmodel

        mv lprnet_fp32_$1b_2core.bmodel $outdir/
    fi
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi

# batch_size=1
gen_mlir 1
gen_fp32bmodel 1

popd