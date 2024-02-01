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
        --model_name depth_processor \
        --model_def ./depth_processor.pt \
        --input_shapes [1,3,384,384] \
        --mlir depth_processor.mlir
}

function gen_fp16bmodel()
{
    model_deploy.py \
        --mlir depth_processor.mlir \
        --quantize BF16 \
        --chip bm1684x \
        --model depth_processor_fp16.bmodel
}

gen_mlir
gen_fp16bmodel

mv depth_processor_fp16.bmodel $outdir/

popd