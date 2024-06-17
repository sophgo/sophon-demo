#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))
outdir=../models/BM1684X/

if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi

function gen_text_encoder_1_mlir()
{
    model_transform.py \
        --model_name te_encoder_1 \
        --model_def ../models/onnx_pt/text_encoder_1/text_encoder_1.onnx \
        --input_shapes [[1,77]] \
        --mlir text_encoder_1.mlir
}

function gen_text_encoder_2_mlir()
{
    model_transform.py \
        --model_name te_encoder_2 \
        --model_def ../models/onnx_pt/text_encoder_2/text_encoder_2.onnx \
        --input_shapes [[1,77]] \
        --mlir text_encoder_2.mlir
}

function gen_text_encoder_1_fp32bmodel()
{
    model_deploy.py \
        --mlir text_encoder_1.mlir \
        --quantize F32 \
        --chip bm1684x \
        --model text_encoder_1_1684x_f32.bmodel

    mv text_encoder_1_1684x_f32.bmodel $outdir/
}

function gen_text_encoder_2_fp16bmodel()
{
    model_deploy.py \
        --mlir text_encoder_2.mlir \
        --quantize F16 \
        --chip bm1684x \
        --model text_encoder_2_1684x_f16.bmodel

    mv text_encoder_2_1684x_f16.bmodel $outdir/
}

pushd $model_dir

gen_text_encoder_1_mlir
gen_text_encoder_1_fp32bmodel

gen_text_encoder_2_mlir
gen_text_encoder_2_fp16bmodel
popd