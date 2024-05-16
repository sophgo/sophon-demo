#!/bin/bash

script_directory="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

pushd "$script_directory/../models/onnx_pt/multilize/vae_encoder" || exit
outdir=../../../BM1684X/multilize/

if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi

for file in *.pt; 
do
    if [[ $file =~ ([0-9]+)_([0-9]+)\.pt ]]; then
        height=${BASH_REMATCH[1]}
        width=${BASH_REMATCH[2]}
    
    model_transform.py \
        --model_name vae_encoder \
        --model_def vae_encoder_${height}_${width}.pt \
        --input_shapes [1,3,${height},${width}] \
        --mlir vae_encoder_${height}_${width}.mlir

    model_deploy.py \
        --mlir vae_encoder_${height}_${width}.mlir \
        --quantize BF16 \
        --chip bm1684x \
        --merge_weight \
        --model vae_encoder_${height}_${width}.bmodel
    fi
done

model_tool \
    --combine *.bmodel \
    -o vae_encoder_multize.bmodel

mv vae_encoder_multize.bmodel $outdir

popd