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

# GLC 拆成编码器/解码器两个子模型，两者之间有一段 CPU 码本查找，
# 因此每个精度都要编译出两个 bmodel。详见 ../docs/GLC_Export_Guide.md
function gen_mlir()
{
    model_transform.py \
        --model_name glc_encoder \
        --model_def ../models/onnx/glc_encoder.onnx \
        --input_shapes "[[1,3,512,768],[1,256,1,1]]" \
        --input_types float32,float32 \
        --mlir glc_encoder.mlir

    model_transform.py \
        --model_name glc_decoder \
        --model_def ../models/onnx/glc_decoder.onnx \
        --input_shapes "[[1,256,32,48],[1,256,8,12],[1,256,1,1]]" \
        --input_types float32,float32,float32 \
        --mlir glc_decoder.mlir
}

function gen_bf16bmodel()
{
    model_deploy.py \
        --mlir glc_encoder.mlir \
        --quantize BF16 \
        --chip $target \
        --model glc_encoder_bf16.bmodel

    model_deploy.py \
        --mlir glc_decoder.mlir \
        --quantize BF16 \
        --chip $target \
        --model glc_decoder_bf16.bmodel

    mv glc_encoder_bf16.bmodel glc_decoder_bf16.bmodel $outdir/
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi

gen_mlir
gen_bf16bmodel

popd
