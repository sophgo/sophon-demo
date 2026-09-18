#!/bin/bash
# CV84X6(bm1684x2) unet bmodel: BF16
model_dir=$(dirname $(readlink -f "$0"))
outdir=../models/CV84X6/

function gen_unet_mlir()
{
    model_transform.py \
        --model_name unet_base \
        --model_def ../models/onnx_pt/unet/unet_base.pt \
        --input_shapes [[2,4,128,128],[1],[2,77,2048],[2,1280],[2,6]] \
        --mlir unet.mlir
}

function gen_unet_bf16bmodel()
{
    model_deploy.py \
        --mlir unet.mlir \
        --quantize BF16 \
        --chip bm1684x2 \
        --model unet_base_cv84x6_bf16.bmodel

    mv unet_base_cv84x6_bf16.bmodel $outdir/
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi

gen_unet_mlir
gen_unet_bf16bmodel

popd
