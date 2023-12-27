#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))
outdir=../../../models/processors/bmodels

function gen_mlir()
{
    model_transform.py \
        --model_name depth_processor \
        --model_def ./depth_processor.pt \
        --input_shapes [1,3,384,384] \
        --mlir depth_processor.mlir
}

function gen_fp16bmodel()
{
    model_deploy.py \
        --mlir depth_processor.mlir \
        --quantize F16 \
        --chip bm1684x \
        --model depth_processor_fp16.bmodel

    mv depth_processor_fp16.bmodel $outdir/
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi

gen_mlir
gen_fp16bmodel

popd