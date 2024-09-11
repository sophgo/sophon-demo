#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))

echo $model_dir

target=bm1688
target_dir=BM1688


outdir=../models/$target_dir


function gen_mlir_image_encoder()
{
    model_transform.py \
        --model_name sam2_encoder \
        --model_def ../models/onnx/sam2_hiera_tiny_encoder.onnx \
        --input_shapes [[1,3,1024,1024]] \
        --mlir sam2_encoder.mlir
}

function gen_fp16bmodel_image_encoder_1core()
{
    model_deploy.py \
        --mlir sam2_encoder.mlir \
        --quantize F16 \
        --chip $target \
        --model sam2_encoder_f16_1b_1core.bmodel

    mv sam2_encoder_f16_1b_1core.bmodel $outdir/image_encoder/
}

function gen_fp16bmodel_image_encoder_2core()
{
    model_deploy.py \
        --mlir sam2_encoder.mlir \
        --quantize F16 \
        --chip $target \
        --num_core 2 \
        --model sam2_encoder_f16_1b_2core.bmodel

    mv sam2_encoder_f16_1b_2core.bmodel $outdir/image_encoder/
}

function gen_mlir_image_decoder()
{
    model_transform.py \
        --model_name sam2_decoder \
        --model_def ../models/onnx/sam2_hiera_tiny_decoder.onnx \
        --input_shapes [[1,256,64,64],[1,32,256,256],[1,64,128,128],[1,1,2],[1,1],[1,1,256,256],[1]] \
        --mlir sam2_decoder.mlir
}

function gen_fp16bmodel_image_decoder_1core()
{
    model_deploy.py \
        --mlir sam2_decoder.mlir \
        --quantize F16 \
        --chip $target \
        --model sam2_decoder_f16_1b_1core.bmodel

    mv sam2_decoder_f16_1b_1core.bmodel $outdir/image_decoder/
}

function gen_fp16bmodel_image_decoder_2core()
{
    model_deploy.py \
        --mlir sam2_decoder.mlir \
        --quantize F16 \
        --chip $target \
        --num_core 2 \
        --model sam2_decoder_f16_1b_2core.bmodel

    mv sam2_decoder_f16_1b_2core.bmodel $outdir/image_decoder/
}

pushd $model_dir
if [ ! -d $outdir/image_encoder ] ; then
    mkdir -p $outdir/image_encoder

else
    echo "Models folder exist! "
fi

if [ ! -d $outdir/image_decoder ] ; then
    mkdir -p $outdir/image_decoder

else
    echo "Models folder exist! "
fi

batch_size=1
gen_mlir_image_encoder 1
gen_fp16bmodel_image_encoder_1core 1
gen_fp16bmodel_image_encoder_2core 1

gen_mlir_image_decoder 1
gen_fp16bmodel_image_decoder_1core 1
gen_fp16bmodel_image_decoder_2core 1

popd