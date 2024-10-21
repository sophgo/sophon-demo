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
        --model_name slowfast \
        --model_def ../models/onnx/slowfast_r50.onnx \
        --input_shapes [[$1,3,8,256,256],[$1,3,32,256,256]] \
        --mlir slowfast_$1b.mlir
}

function gen_cali_table()
{
    run_calibration.py slowfast_$1b.mlir \
        --dataset ../datasets/cali_set_npy \
        --input_num 128 \
        -o slowfast_cali_table
}

function gen_int8bmodel()
{
    model_deploy.py \
        --mlir slowfast_$1b.mlir \
        --quantize INT8 \
        --chip $target \
        --calibration_table slowfast_cali_table \
        --model slowfast_${target}_int8_$1b.bmodel
    mv slowfast_${target}_int8_$1b.bmodel $outdir
    if test $target = "bm1688";then
        model_deploy.py \
            --mlir slowfast_$1b.mlir \
            --quantize INT8 \
            --chip $target \
            --calibration_table slowfast_cali_table \
            --model slowfast_${target}_int8_$1b_2core.bmodel \
            --num_core 2
        mv slowfast_${target}_int8_$1b_2core.bmodel $outdir
    fi
}

pushd $model_dir
if [ ! -d "$outdir" ]; then
    echo $pwd
    mkdir $outdir
fi

cd ../tools/
python3 slowfast_npy.py
cd ../scripts

# batch_size=1
gen_mlir 1
gen_cali_table 1
gen_int8bmodel 1

# batch_size=4
gen_mlir 4
gen_int8bmodel 4
popd
