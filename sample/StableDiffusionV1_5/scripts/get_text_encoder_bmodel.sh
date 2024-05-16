#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))
outdir1=../models/BM1684X/singlize/
outdir2=../models/BM1684X/multilize/

if [ ! -d $outdir1 ]; then
    mkdir -p $outdir1
fi

if [ ! -d $outdir2 ]; then
    mkdir -p $outdir2
fi

function gen_text_encoder_mlir()
{
    model_transform.py \
        --model_name encoder \
        --model_def ../models/onnx_pt/text_encoder_1684x_f32.onnx \
        --input_shapes [[1,77]] \
        --mlir encoder.mlir
}

function gen_text_encoder_fp32bmodel()
{
    model_deploy.py \
        --mlir encoder.mlir \
        --quantize F32 \
        --chip bm1684x \
        --model text_encoder_1684x_f32.bmodel

    cp text_encoder_1684x_f32.bmodel $outdir1/
    mv text_encoder_1684x_f32.bmodel $outdir2/
}

pushd $model_dir

gen_text_encoder_mlir
gen_text_encoder_fp32bmodel

popd