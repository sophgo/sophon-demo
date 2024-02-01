#!/bin/bash

script_directory="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
pushd "$script_directory/processors" || exit

outdir=../../../../models/BM1684X/processors/

if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi

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
        --quantize BF16 \
        --chip bm1684x \
        --model segmentation_processor_fp16.bmodel

    mv segmentation_processor_fp16.bmodel $outdir/
}

gen_mlir
gen_fp16bmodel

popd