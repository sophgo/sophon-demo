#!/bin/bash
# CV84X6(bm1684x2) vae_decoder: BF16, 输入 [[1,4,64,64]]
model_dir=$(dirname $(readlink -f "$0"))
outdir=../models/CV84X6/singlize/

if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi

function gen_vae_decoder_mlir()
{
    model_transform.py \
        --model_name vae_decoder \
        --model_def ../models/onnx_pt/singlize/vae_decoder_singlize.pt \
        --input_shapes [[1,4,64,64]] \
        --mlir vae_decoder.mlir
}

function gen_vae_decoder_bf16bmodel()
{
    model_deploy.py \
        --mlir vae_decoder.mlir \
        --quantize BF16 \
        --chip bm1684x2 \
        --model vae_decoder_cv84x6_bf16.bmodel

    mv vae_decoder_cv84x6_bf16.bmodel $outdir/
}

pushd $model_dir

gen_vae_decoder_mlir
gen_vae_decoder_bf16bmodel

popd
