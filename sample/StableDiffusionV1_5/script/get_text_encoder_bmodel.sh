#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))
outdir=../models/BM1684X/singlize/

function gen_mlir()
{
    model_transform.py \
        --model_name encoder \
        --model_def ../models/onnx_pt/text_encoder_1684x_f32.onnx \
        --input_shapes [[1,77]] \
        --mlir encoder.mlir
}

function gen_fp32bmodel()
{
    model_deploy.py \
        --mlir encoder.mlir \
        --quantize F32 \
        --chip bm1684x \
        --model text_encoder_1684x_f32.bmodel

    mv text_encoder_1684x_f32.bmodel $outdir/
}

pushd $model_dir

if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi

gen_mlir
gen_fp32bmodel

popd