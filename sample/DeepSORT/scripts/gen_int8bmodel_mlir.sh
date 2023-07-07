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
        --model_name extractor \
        --model_def ../models/onnx/extractor.onnx \
        --input_shapes [[$1,3,128,64]] \
        --mean 2.1179039,1.9912664,1.772926 \
        --scale 0.0171248,0.017507,0.0174292 \
        --pixel_format rgb  \
        --mlir extractor_$1b.mlir
}

function gen_cali_table()
{
    run_calibration.py extractor_$1b.mlir \
        --dataset ../datasets/cali_set/ \
        --input_num 128 \
        -o extractor_cali_table
}

function gen_int8bmodel()
{
    model_deploy.py \
        --mlir extractor_$1b.mlir \
        --quantize INT8 \
        --chip $target \
        --calibration_table extractor_cali_table \
        --model extractor_int8_$1b.bmodel

    mv extractor_int8_$1b.bmodel $outdir/
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