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
        --model_name yolov4_416 \
        --model_def ../data/models/onnx/yolov4_1_3_416_416_static.onnx \
        --input_shapes [[$1,3,416,416]] \
        --mean 0.0,0.0,0.0 \
        --scale 0.0039216,0.0039216,0.0039216 \
        --test_input ../data/images/dog.jpg \
        --test_result tmp.npz \
        --output_names /models.149/conv102/Conv_output_0,/models.160/conv110/Conv_output_0,/models.138/conv94/Conv_output_0 \
        --mlir yolov4_416_$1b.mlir
}

function gen_cali_table()
{
    run_calibration.py yolov4_416_$1b.mlir \
        --dataset ../datasets/coco128/ \
        --input_num 128 \
        -o yolov4_cali_table
}

function gen_int8bmodel()
{
    model_deploy.py \
        --mlir yolov4_416_$1b.mlir \
        --calibration_table yolov4_cali_table \
        --quantize INT8 \
        --chip $target \
        --test_input ../data/images/dog.jpg \
        --test_reference tmp.npz \
        --model yolov4_416_int8_$1b.bmodel

    mv yolov4_416_int8_$1b.bmodel $outdir/
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
batch_size=1
gen_mlir 1
gen_cali_table 1
gen_int8bmodel 1

#batch_size=4
#gen_mlir 4
#gen_cali_table 4
#gen_int8bmodel 4

popd