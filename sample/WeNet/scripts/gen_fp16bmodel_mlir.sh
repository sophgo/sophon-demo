#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))

if [ ! $1 ]; then
    target=bm1684x
    target_dir=BM1684X
else
    target=${1,,}
    target_dir=${target^^}
    if test $target = "bm1684"
    then
        echo "bm1684 do not support fp16"
        exit
    fi
fi
outdir=../models/$target_dir

function gen_mlir_encoder()
{
    model_transform.py \
        --model_name wenet_encoder \
        --model_def ../models/onnx/wenet_encoder.onnx \
        --input_shapes [[1,67,80],[1],[1,1],[1,12,4,80,128],[1,12,256,7],[1,1,80]] \
        --mlir wenet_encoder.mlir
}

function gen_fp16bmodel_encoder()
{
    model_deploy.py \
        --mlir wenet_encoder.mlir \
        --quantize F16 \
        --chip $target \
        --model wenet_encoder_fp16.bmodel

    mv wenet_encoder_fp16.bmodel $outdir/
}


function gen_mlir_decoder()
{
    model_transform.py \
        --model_name wenet_decoder \
        --model_def ../models/onnx/wenet_decoder.onnx \
        --input_shapes [[1,350,256],[1],[1,10,350],[1,10],[1,10,350],[1,10]] \
        --mlir wenet_decoder.mlir
        # --test_input wenet_decoder_input.npz \
        # --test_result wenet_decoder_top.npz
}

function gen_fp16bmodel_decoder()
{
    model_deploy.py \
        --mlir wenet_decoder.mlir \
        --quantize F16 \
        --chip $target \
        --model wenet_decoder_fp16.bmodel \
        --compare_all \
        --quantize_table ../models/onnx/wenet_decoder_qtable \
        --tolerance 0.99,0.99
        # --test_input wenet_decoder_input.npz \
        # --test_reference wenet_decoder_top.npz \

    mv wenet_decoder_fp16.bmodel $outdir/
}


pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
gen_mlir_encoder
gen_fp16bmodel_encoder
gen_mlir_decoder
gen_fp16bmodel_decoder
popd