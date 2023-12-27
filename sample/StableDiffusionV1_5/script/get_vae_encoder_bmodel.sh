#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))
outdir=../models/BM1684X/singlize/

function gen_mlir()
{
    model_transform.py \
        --model_name vae_encoder \
        --model_def ../models/onnx_pt/vae_encoder_singlize.pt \
        --input_shapes [[1,3,512,512]] \
        --mlir vae_encoder.mlir
}

function gen_fp16bmodel()
{
    model_deploy.py \
        --mlir vae_encoder.mlir \
        --quantize F16 \
        --chip bm1684x \
        --model vae_encoder_1684x_f16.bmodel

    mv vae_encoder_1684x_f16.bmodel $outdir/
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi

gen_mlir
gen_fp16bmodel

popd