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
        --model_name yolov8s \
        --model_def ../models/onnx/yolov8s_getmask_32_fp32.onnx \
        --input_shapes [[1,32,32],[1,32,160,160]] \
        --mlir yolov8s_getmask_32.mlir \
        --dynamic_shape_input_names mask_info
}

function gen_fp32bmodel()
{
    model_deploy.py \
        --mlir yolov8s_getmask_32.mlir \
        --quantize F32 \
        --chip $target \
        --model yolov8s_getmask_32_fp32.bmodel \
        --dynamic
    mv yolov8s_getmask_32_fp32.bmodel $outdir/
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi

gen_mlir 
gen_fp32bmodel 

popd
