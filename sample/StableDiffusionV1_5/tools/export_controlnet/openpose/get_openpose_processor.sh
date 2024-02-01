#!/bin/bash

script_directory="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
pushd "$script_directory/processors" || exit

outdir=../../../../models/BM1684X/processors/

if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi


mv openpose_hand_processor.pt ..
mv openpose_face_processor.pt ..

for file in *.pt; 
do
    if [[ $file =~ ([0-9]+)_([0-9]+)\.pt ]]; then
        height=${BASH_REMATCH[1]}
        width=${BASH_REMATCH[2]}
    
    model_transform.py \
        --model_name openpose_body_processor \
        --model_def openpose_body_processor_${height}_${width}.pt \
        --input_shapes [1,3,${height},${width}] \
        --mlir openpose_body_processor_${height}_${width}.mlir

    model_deploy.py \
        --mlir openpose_body_processor_${height}_${width}.mlir \
        --quantize BF16 \
        --chip bm1684x \
        --merge_weight \
        --model openpose_body_processor_${height}_${width}.bmodel
    fi
done

model_tool \
    --combine *.bmodel \
    -o openpose_body_processor_fp16.bmodel

mv openpose_body_processor_fp16.bmodel $outdir

mv ../openpose_hand_processor.pt .
mv ../openpose_face_processor.pt .

model_transform.py \
--model_name openpose_hand \
--model_def openpose_hand_processor.pt \
--input_shapes [[1,3,184,184]] \
--mlir openpose_hand.mlir

model_deploy.py \
--mlir openpose_hand.mlir \
--quantize F16 \
--chip bm1684x \
--model openpose_hand_processor_fp16.bmodel


model_transform.py \
--model_name openpose_face \
--model_def openpose_face_processor.pt \
--input_shapes [[1,3,384,384]] \
--mlir openpose_face.mlir

model_deploy.py \
--mlir openpose_face.mlir \
--quantize F16 \
--chip bm1684x \
--model openpose_face_processor_fp16.bmodel


mv openpose_hand_processor_fp16.bmodel $outdir
mv openpose_face_processor_fp16.bmodel $outdir

popd