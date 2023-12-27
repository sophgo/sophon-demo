#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))
outdir=../../../models/processors/bmodels

function gen_mlir()
{
    model_transform.py \
        --model_name segmentation_processor \
        --model_def ./segmentation_processor.onnx \
        --input_shapes [1,3,576,576] \
        --mlir segmentation_processor.mlir
}

function gen_fp16bmodel()
{
    model_deploy.py \
        --mlir segmentation_processor.mlir \
        --quantize F16 \
        --chip bm1684x \
        --model segmentation_processor_fp16.bmodel

    mv segmentation_processor_fp16.bmodel $outdir/
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi

gen_mlir
gen_fp16bmodel

popd