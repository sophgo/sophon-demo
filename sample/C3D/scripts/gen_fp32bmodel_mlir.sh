#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))

if [ ! $1 ]; then
    target=bm1684x
else
    target=$1
fi

outdir=../models/BM1684X

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
        --model $outdir/c3d_fp32_$1b.bmodel
}

pushd $model_dir
if [ ! -d "$outdir" ]; then
    echo $pwd
    mkdir $outdir
fi

# batch_size=1
gen_mlir 1
gen_fp32bmodel 1
rm c3d* final_opt.onnx

# batch_size=4
gen_mlir 4
gen_fp32bmodel 4
rm c3d* final_opt.onnx
popd