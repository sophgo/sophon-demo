#!/bin/bash

script_directory="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

pushd "$script_directory/../models/onnx_pt/multilize/unet" || exit
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
        --model_name unet \
        --model_def unet_${height}_${width}.pt \
        --input_shapes [[2,4,$((height/8)),$((width/8))],[1],[2,77,768],[2,1280,$((height/64)),$((width/64))],[2,320,$((height/8)),$((width/8))],[2,320,$((height/8)),$((width/8))],[2,320,$((height/8)),$((width/8))],[2,320,$((height/16)),$((width/16))],[2,640,$((height/16)),$((width/16))],[2,640,$((height/16)),$((width/16))],[2,640,$((height/32)),$((width/32))],[2,1280,$((height/32)),$((width/32))],[2,1280,$((height/32)),$((width/32))],[2,1280,$((height/64)),$((width/64))],[2,1280,$((height/64)),$((width/64))],[2,1280,$((height/64)),$((width/64))]] \
        --mlir unet_${height}_${width}.mlir

    model_deploy.py \
        --mlir unet_${height}_${width}.mlir \
        --quantize BF16 \
        --chip bm1684x \
        --merge_weight \
        --model unet_${height}_${width}.bmodel
    fi
done

model_tool \
    --combine *.bmodel \
    -o unet_multize.bmodel

mv unet_multize.bmodel $outdir

popd