#!/bin/bash

script_directory="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
pushd "$script_directory/controlnets" || exit

outdir=../../../../models/BM1684X/controlnets/

if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi

for file in *.pt; 
do
    if [[ $file =~ ([0-9]+)_([0-9]+)\.pt ]]; then
        height=${BASH_REMATCH[1]}
        width=${BASH_REMATCH[2]}
        latent_height=$((height/8))
        latent_width=$((width/8))
    
    model_transform.py \
        --model_name openpose_controlnet \
        --model_def openpose_controlnet_${height}_${width}.pt \
        --input_shapes [[2,4,${latent_height},${latent_width}],[1],[2,77,768],[2,3,${height},${width}]] \
        --mlir openpose_controlnet_${height}_${width}.mlir

    model_deploy.py \
        --mlir openpose_controlnet_${height}_${width}.mlir \
        --quantize BF16 \
        --chip bm1684x \
        --merge_weight \
        --model openpose_controlnet_${height}_${width}.bmodel
    fi
done

model_tool \
    --combine *.bmodel \
    -o openpose_controlnet_fp16.bmodel

mv openpose_controlnet_fp16.bmodel $outdir

popd