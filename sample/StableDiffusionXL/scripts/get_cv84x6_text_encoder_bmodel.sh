#!/bin/bash
# CV84X6(bm1684x2) bmodels: text_encoder_1/2 均为 F16
# 注意: CV84X6 当前固件 F32 的 matmul/fc 会命中 mm1 断言, text_encoder 只能编 F16
model_dir=$(dirname $(readlink -f "$0"))
outdir=../models/CV84X6/

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

function gen_text_encoder_1_f16bmodel()
{
    model_deploy.py \
        --mlir text_encoder_1.mlir \
        --quantize F16 \
        --chip bm1684x2 \
        --model text_encoder_1_cv84x6_f16.bmodel

    mv text_encoder_1_cv84x6_f16.bmodel $outdir/
}

function gen_text_encoder_2_f16bmodel()
{
    model_deploy.py \
        --mlir text_encoder_2.mlir \
        --quantize F16 \
        --chip bm1684x2 \
        --model text_encoder_2_cv84x6_f16.bmodel

    mv text_encoder_2_cv84x6_f16.bmodel $outdir/
}

pushd $model_dir

gen_text_encoder_1_mlir
gen_text_encoder_1_f16bmodel

gen_text_encoder_2_mlir
gen_text_encoder_2_f16bmodel
popd
