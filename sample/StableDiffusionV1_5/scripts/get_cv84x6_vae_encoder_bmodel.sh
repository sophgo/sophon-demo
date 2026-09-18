#!/bin/bash
# CV84X6(bm1684x2) vae_encoder: BF16, 输入 [[1,3,512,512]]
model_dir=$(dirname $(readlink -f "$0"))
outdir=../models/CV84X6/singlize/

if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi

function gen_vae_encoder_mlir()
{
    model_transform.py \
        --model_name vae_encoder \
        --model_def ../models/onnx_pt/singlize/vae_encoder_singlize.pt \
        --input_shapes [[1,3,512,512]] \
        --mlir vae_encoder.mlir
}

function gen_vae_encoder_bf16bmodel()
{
    model_deploy.py \
        --mlir vae_encoder.mlir \
        --quantize BF16 \
        --chip bm1684x2 \
        --model vae_encoder_cv84x6_bf16.bmodel

    mv vae_encoder_cv84x6_bf16.bmodel $outdir/
}

pushd $model_dir

gen_vae_encoder_mlir
gen_vae_encoder_bf16bmodel

popd
