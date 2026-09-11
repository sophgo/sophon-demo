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
        echo "bm1684 do not support int8"
        exit
    fi
fi
outdir=../models/$target_dir

function gen_mlir_encoder()
{
    model_transform.py \
        --model_name wenet_encoder_streaming \
        --model_def ../models/onnx/wenet_encoder_streaming.onnx \
        --input_shapes [[1,67,80],[1],[1,1],[1,12,4,80,128],[1,12,256,7],[1,1,80]] \
        --mlir wenet_encoder_streaming.mlir

    model_transform.py \
        --model_name wenet_encoder_non_streaming \
        --model_def ../models/onnx/wenet_encoder_non_streaming.onnx \
        --input_shapes [[1,1200,80],[1]] \
        --mlir wenet_encoder_non_streaming.mlir
}

function gen_cali_table_encoder()
{
    run_calibration.py wenet_encoder_streaming.mlir \
        --dataset ../datasets/cali_npz/stream/ \
        --input_num 100 \
        --cali_method mse \
        -o wenet_encoder_streaming_cali_table

    run_calibration.py wenet_encoder_non_streaming.mlir \
        --dataset ../datasets/cali_npz/nonstream/ \
        --input_num 100 \
        --cali_method percentile9999 \
        -o wenet_encoder_non_streaming_cali_table
}

function gen_int8bmodel_encoder()
{
    model_deploy.py \
        --mlir wenet_encoder_streaming.mlir \
        --quantize INT8 \
        --chip $target \
        --calibration_table wenet_encoder_streaming_cali_table \
        --quantize_table wenet_encoder_streaming_qtable \
        --model wenet_encoder_streaming_int8.bmodel

    mv wenet_encoder_streaming_int8.bmodel $outdir/

    model_deploy.py \
        --mlir wenet_encoder_non_streaming.mlir \
        --quantize INT8 \
        --chip $target \
        --calibration_table wenet_encoder_non_streaming_cali_table \
        --quantize_table wenet_encoder_non_streaming_qtable \
        --model wenet_encoder_non_streaming_int8.bmodel

    mv wenet_encoder_non_streaming_int8.bmodel $outdir/
}

function gen_mlir_decoder()
{
    model_transform.py \
        --model_name wenet_decoder \
        --model_def ../models/onnx/wenet_decoder.onnx \
        --input_shapes [[1,350,256],[1],[1,10,350],[1,10],[1,10,350],[1,10]] \
        --mlir wenet_decoder.mlir
}

function gen_cali_table_decoder()
{
    run_calibration.py wenet_decoder.mlir \
        --dataset ../datasets/cali_npz/decoder/ \
        --input_num 96 \
        --cali_method mse \
        -o wenet_decoder_cali_table
}

function gen_int8bmodel_decoder()
{
    model_deploy.py \
        --mlir wenet_decoder.mlir \
        --quantize INT8 \
        --chip $target \
        --model wenet_decoder_int8.bmodel \
        --calibration_table wenet_decoder_cali_table \
        --quantize_table wenet_decoder_qtable

    mv wenet_decoder_int8.bmodel $outdir/
}


pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
gen_mlir_encoder
gen_cali_table_encoder
gen_int8bmodel_encoder
gen_mlir_decoder
gen_cali_table_decoder
gen_int8bmodel_decoder
popd