#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))

if [ ! $1 ]; then
    target=bm1684x
    target_dir=BM1684X
else
    target=${1,,}
    target_dir=${target^^}
fi

outdir=../data/models/$target_dir

function gen_mlir()
{
    model_transform.py \
        --model_name yolov4_416_coco \
        --model_def ../data/models/onnx/yolov4_1_3_416_416_static.onnx \
        --input_shapes [[$1,3,416,416]] \
        --mean 0.0,0.0,0.0 \
        --scale 0.0039216,0.0039216,0.0039216 \
        --test_input ../data/images/dog.jpg \
        --test_result tmp.npz \
        --output_names /models.149/conv102/Conv_output_0,/models.160/conv110/Conv_output_0,/models.138/conv94/Conv_output_0 \
        --mlir yolov4_416_$1b.mlir
}

function gen_fp32bmodel()
{
    model_deploy.py \
        --mlir yolov4_416_$1b.mlir \
        --quantize F32 \
        --chip $target \
        --test_input ../data/images/dog.jpg \
        --test_reference tmp.npz \
        --model yolov4_416_fp32_$1b.bmodel

    mv yolov4_416_fp32_$1b.bmodel $outdir/
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
gen_mlir 1
gen_fp32bmodel 1

popd