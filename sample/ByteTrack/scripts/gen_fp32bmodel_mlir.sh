#!/bin/bash
if [ ! $1 ]; then
    target=bm1684x
else
    target=$1
fi

outdir=../models/BM1684X

function gen_mlir()
{
    model_transform.py \
        --model_name bytetrack_s \
        --model_def ../models/onnx/bytetrack_s.onnx \
        --input_shapes [[$1,3,608,1088]] \
        --mlir bytetrack_s_$1b.mlir
}

function gen_fp32bmodel()
{
    model_deploy.py \
        --mlir bytetrack_s_$1b.mlir \
        --quantize F32 \
        --chip $target \
        --model bytetrack_s_fp32_$1b.bmodel

    mv bytetrack_s_fp32_$1b.bmodel $outdir/
}

if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
gen_mlir 1
gen_fp32bmodel 1
