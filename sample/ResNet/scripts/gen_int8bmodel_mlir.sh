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
        --model_name resnet50_$1b \
        --model_def ../models/onnx/resnet50_$1b.onnx \
        --input_shapes [[$1,3,224,224]] \
        --mean 103.53,116.28,123.675 \
        --scale 0.01742919,0.017507,0.01712475 \
        --keep_aspect_ratio \
        --pixel_format rgb  \
        --mlir resnet50_$1b.mlir
}

function gen_cali_table()
{
    run_calibration.py resnet50_$1b.mlir \
        --dataset ../datasets/cali_data/ \
        --input_num 200 \
        -o resnet50_cali_table
}

function gen_int8bmodel()
{
    model_deploy.py \
        --mlir resnet50_$1b.mlir \
        --quantize INT8 \
        --chip $target \
        --calibration_table resnet50_cali_table \
        --asymmetric \
        --model resnet50_int8_$1b.bmodel

    mv resnet50_int8_$1b.bmodel $outdir/
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
gen_mlir 1
gen_cali_table 1
gen_int8bmodel 1

# batch_size=4
gen_mlir 4
gen_int8bmodel 4

popd