#!/bin/bash
model_dir=$(cd `dirname $BASH_SOURCE[0]`/ && pwd)

echo model_dir is $model_dir

if [ ! $1 ]; then
    target=bm1684x
else
    target=$1
fi

outdir=../models/BM1684X

function gen_mlir()
{
    model_transform.py \
        --model_name yolov7_v0.1_3output \
        --model_def ../models/onnx/yolov7_v0.1_3output_$1b.onnx \
        --input_shapes [[$1,3,640,640]] \
        --mlir yolov7_v0.1_3output_$1b.mlir
}

function gen_fp16bmodel()
{
    model_deploy.py \
        --mlir yolov7_v0.1_3output_$1b.mlir \
        --quantize BF16 \
        --chip $target \
        --model yolov7_v0.1_3output_fp16_$1b.bmodel

    mv yolov7_v0.1_3output_fp16_$1b.bmodel $outdir/
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
gen_mlir 1
gen_fp16bmodel 1
# batch_size=4
gen_mlir 4
gen_fp16bmodel 4
popd