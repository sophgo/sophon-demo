#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))

if [ ! $1 ]; then
    target=bm1684x
    target_dir=BM1684X
else
    target=${1,,}
    target_dir=${target^^}
fi

outdir=../models/$target_dir

function gen_mlir()
{   
    model_transform.py \
        --model_name yolov8s_1output \
        --model_def ../models/onnx/yolov8s_$1b.onnx \
        --input_shapes [[$1,3,640,640]] \
        --mean 0.0,0.0,0.0 \
        --scale 0.0039216,0.0039216,0.0039216 \
        --keep_aspect_ratio \
        --pixel_format rgb  \
        --mlir yolov8s_$1b.mlir
}

function gen_fp16bmodel()
{
    model_deploy.py \
        --mlir yolov8s_$1b.mlir \
        --quantize F32 \
        --chip $target \
        --model yolov8s_fp32_$1b.bmodel

    mv yolov8s_fp32_$1b.bmodel $outdir/
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
gen_mlir 1
gen_fp16bmodel 1

popd