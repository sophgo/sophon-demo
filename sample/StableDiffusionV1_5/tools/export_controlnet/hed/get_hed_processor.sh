#!/bin/bash

script_directory="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
pushd "$script_directory/processors" || exit

outdir=../../../../models/BM1684X/processors/
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi

for file in *.pt; 
do
    if [[ $file =~ ([0-9]+)_([0-9]+)\.pt ]]; then
        height=${BASH_REMATCH[1]}
        width=${BASH_REMATCH[2]}
    
    model_transform.py \
        --model_name hed_processor \
        --model_def hed_processor_${height}_${width}.pt \
        --input_shapes [1,3,${height},${width}] \
        --mlir hed_processor_${height}_${width}.mlir

    model_deploy.py \
        --mlir hed_processor_${height}_${width}.mlir \
        --quantize BF16 \
        --chip bm1684x \
        --merge_weight \
        --model hed_processor_${height}_${width}.bmodel
    fi
done 

model_tool \
    --combine *.bmodel \
    -o hed_processor_fp16.bmodel

mv hed_processor_fp16.bmodel $outdir

popd