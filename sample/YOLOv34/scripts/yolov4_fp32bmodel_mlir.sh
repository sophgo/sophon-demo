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
        --model_name yolov4_416_coco \
        --model_def ../models/onnx/yolov4_$1b.onnx \
        --input_shapes [[$1,3,416,416]] \
        --mean 0.0,0.0,0.0 \
        --scale 0.0039216,0.0039216,0.0039216 \
        --test_input ../datasets/test/dog.jpg \
        --test_result tmp.npz \
        --mlir yolov4_416_$1b.mlir
}

function gen_fp32bmodel()
{
    model_deploy.py \
        --mlir yolov4_416_$1b.mlir \
        --quantize F32 \
        --chip $target \
        --test_input ../datasets/test/dog.jpg \
        --test_reference tmp.npz \
        --model yolov4_fp32_$1b.bmodel

    mv yolov4_fp32_$1b.bmodel $outdir/
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
gen_mlir 1
gen_fp32bmodel 1

popd