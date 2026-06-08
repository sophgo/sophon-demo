#!/bin/bash
# INT8 BModel compilation script for ArcFace ResNet50
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
        --model_name arcface_resnet50 \
        --model_def ../models/onnx/w600k_r50.onnx \
        --input_shapes [[$1,3,112,112]] \
        --mean 127.5,127.5,127.5 \
        --scale 0.0078125,0.0078125,0.0078125 \
        --pixel_format rgb \
        --mlir arcface_resnet50_$1b.mlir
}

function gen_cali_table()
{
    run_calibration.py arcface_resnet50_$1b.mlir \
        --dataset ../datasets/cali \
        --input_num 100 \
        -o arcface_cali_table_$1b
}

function gen_int8bmodel()
{
    model_deploy.py \
        --mlir arcface_resnet50_$1b.mlir \
        --quantize INT8 \
        --chip $target \
        --calibration_table arcface_cali_table_$1b \
        --model arcface_resnet50_int8_$1b.bmodel

    mv arcface_resnet50_int8_$1b.bmodel $outdir/
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
gen_cali_table 4
gen_int8bmodel 4

popd
