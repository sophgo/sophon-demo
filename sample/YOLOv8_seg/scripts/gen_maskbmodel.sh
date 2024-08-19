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
        --model_def ../models/onnx/yolov8_getmask_32.onnx \
        --input_shapes [[1,32,32],[1,32,160,160]] \
        --mlir yolov8s_getmask_32.mlir \
        --dynamic_shape_input_names mask_info
}

function gen_int8bmodel()
{
    model_deploy.py     \
        --mlir yolov8s_getmask_32.mlir  \
        --quantize INT8     \
        --chip $target    \
        --calibration_table ../models/onnx/yolov8s_getmask_32_cali_table     \
        --quantize_table ../models/onnx/yolov8s_getmask_qtable     \
        --model yolov8s_getmask_32_int8.bmodel     \
        --quant_output     \
        --dynamic

    mv yolov8s_getmask_32_int8.bmodel  $outdir/

    if test $target = "bm1688";then
        model_deploy.py \
            --mlir yolov8s_getmask_32.mlir  \
            --quantize INT8     \
            --chip $target    \
            --calibration_table ../models/onnx/yolov8s_getmask_32_cali_table     \
            --quantize_table ../models/onnx/yolov8s_getmask_qtable     \
            --model yolov8s_getmask_32_int8_2core.bmodel     \
            --quant_output   \
            --num_core 2 \
            --dynamic

        mv yolov8s_getmask_32_int8_2core.bmodel $outdir/
    fi
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi

gen_mlir 
gen_int8bmodel

popd