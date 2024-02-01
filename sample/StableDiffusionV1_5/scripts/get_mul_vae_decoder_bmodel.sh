#!/bin/bash

script_directory="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

pushd "$script_directory/../models/onnx_pt/multilize/vae_decoder" || exit
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
        --model_name vae_decoder \
        --model_def vae_decoder_${height}_${width}.pt \
        --input_shapes [1,4,$((height/8)),$((width/8))] \
        --mlir vae_decoder_${height}_${width}.mlir

    model_deploy.py \
        --mlir vae_decoder_${height}_${width}.mlir \
        --quantize BF16 \
        --chip bm1684x \
        --merge_weight \
        --model vae_decoder_${height}_${width}.bmodel
    fi
done

model_tool \
    --combine *.bmodel \
    -o vae_decoder_multize.bmodel

mv vae_decoder_multize.bmodel $outdir

popd