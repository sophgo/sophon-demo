#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))

if [ ! $1 ]; then
    target=bm1684x
    target_dir=BM1684X
else
    target=${1,,}
    target_dir=${target^^}
    if test $target = "bm1684"
    then
        echo "bm1684 do not support fp16"
        exit
    fi
fi

outdir=../models/$target_dir
function gen_mlir()
{   
    onnx_path=../models/onnx/yolov8s.onnx
    if test $model_name = "yolov8s"; then
        onnx_path=../models/onnx/yolov8s.onnx
    elif test $model_name = "yolov9s"; then
        onnx_path=../models/onnx/yolov9-s-converted.onnx
    elif test $model_name = "yolov11s"; then
        onnx_path=../models/onnx/yolo11s.onnx
    elif test $model_name = "yolov12s"; then
        onnx_path=../models/onnx/yolov12s.onnx
    fi
    model_transform.py \
        --model_name ${model_name} \
        --model_def $onnx_path \
        --input_shapes [[$1,3,640,640]] \
        --mean 0.0,0.0,0.0 \
        --scale 0.0039216,0.0039216,0.0039216 \
        --keep_aspect_ratio \
        --pixel_format rgb  \
        --mlir ${model_name}_$1b.mlir
}

function gen_fp16bmodel()
{
    model_deploy.py \
        --mlir ${model_name}_$1b.mlir \
        --quantize F16 \
        --chip $target \
        --model ${model_name}_fp16_$1b.bmodel

    mv ${model_name}_fp16_$1b.bmodel $outdir/
    if test $target = "bm1688";then
        model_deploy.py \
            --mlir ${model_name}_$1b.mlir \
            --quantize F16 \
            --chip $target \
            --model ${model_name}_fp16_$1b_2core.bmodel \
            --num_core 2

        mv ${model_name}_fp16_$1b_2core.bmodel $outdir/
    fi
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
model_name=yolov8s
gen_mlir 1
gen_fp16bmodel 1

model_name=yolov9s
gen_mlir 1
gen_fp16bmodel 1

model_name=yolov11s
gen_mlir 1
gen_fp16bmodel 1

model_name=yolov12s
gen_mlir 1
gen_fp16bmodel 1
popd
