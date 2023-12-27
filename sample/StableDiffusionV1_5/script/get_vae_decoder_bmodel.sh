#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))
outdir=../models/BM1684X/singlize/

function gen_mlir()
{
    model_transform.py \
        --model_name vae_decoder \
        --model_def ../models/onnx_pt/vae_decoder.pt \
        --input_shapes [[1,4,64,64]] \
        --mlir vae_decoder.mlir
}

function gen_fp16bmodel()
{
    model_deploy.py \
        --mlir vae_decoder.mlir \
        --quantize F16 \
        --chip bm1684x \
        --model vae_decoder_1684x_f16.bmodel

    mv vae_decoder_1684x_f16.bmodel $outdir/
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi

gen_mlir
gen_fp16bmodel

popd