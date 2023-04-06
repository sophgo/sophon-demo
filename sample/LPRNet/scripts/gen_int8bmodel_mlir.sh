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
        --model_name lprnet \
        --model_def ../models/onnx/lprnet_$1b.onnx \
        --input_shapes [[$1,3,24,94]] \
        --mean 127.5,127.5,127.5 \
        --scale 0.0078125,0.0078125,0.0078125 \
        --keep_aspect_ratio \
        --pixel_format rgb  \
        --mlir lprnet_$1b.mlir
}

function gen_cali_table()
{
    run_calibration.py lprnet_$1b.mlir \
        --dataset ../datasets/test_md5/ \
        --input_num 1000 \
        -o lprnet_cali_table
}

function gen_int8bmodel()
{
    model_deploy.py \
        --mlir lprnet_$1b.mlir \
        --quantize INT8 \
        --chip $target \
        --calibration_table lprnet_cali_table \
        --asymmetric \
        --model lprnet_int8_$1b.bmodel

    mv lprnet_int8_$1b.bmodel $outdir/
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