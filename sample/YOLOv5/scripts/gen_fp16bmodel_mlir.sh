#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))

if [ ! $1 ]; then
    target=bm1684x
else
    target=$1
fi

outdir=../models/BM1684X

function gen_mlir()
{
    model_transform.py \
        --model_name yolov5s_v6.1_3output \
        --model_def ../models/onnx/yolov5s_v6.1_3output.onnx \
        --input_shapes [[$1,3,640,640]] \
        --mlir yolov5s_v6.1_3output_$1b.mlir
}

function gen_fp16bmodel()
{
    model_deploy.py \
        --mlir yolov5s_v6.1_3output_$1b.mlir \
        --quantize F16 \
        --chip $target \
        --model yolov5s_v6.1_3output_fp16_$1b.bmodel

    mv yolov5s_v6.1_3output_fp16_$1b.bmodel $outdir/
}

pushd $model_dir
# batch_size=1
gen_mlir 1
gen_fp16bmodel 1

popd