#!/bin/bash
# CV84X6(bm1684x2) unet: BF16, 输入 [[2,4,64,64],[1],[2,77,768]]
model_dir=$(dirname $(readlink -f "$0"))
outdir=../models/CV84X6/singlize/

if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
size=768
b=2
if [ $1 == 'sd_turbo' ]; then
    size=1024
    b=1
fi
echo $b

function gen_unet_mlir()
{
    model_transform.py \
        --model_name unet \
        --model_def ../models/onnx_pt/singlize/unet_fp32.pt \
        --input_shapes [[$b,4,64,64],[1],[$b,77,$size]] \
        --mlir unet.mlir
}

function gen_unet_bf16bmodel()
{
    model_deploy.py \
        --mlir unet.mlir \
        --quantize BF16 \
        --chip bm1684x2 \
        --model unet_cv84x6_bf16.bmodel

    mv unet_cv84x6_bf16.bmodel $outdir/
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi

gen_unet_mlir
gen_unet_bf16bmodel
popd
