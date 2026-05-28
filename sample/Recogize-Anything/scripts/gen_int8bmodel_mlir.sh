#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))

if [ ! $1 ]; then
    target=bm1684x
    target_dir=BM1684X
else
    target=${1,,}
    target_dir=${target^^}
    if test $target = "bm1684"
    then
        echo "bm1684 do not support fp16 which is required by mixed precision quantization"
        exit
    fi
fi

outdir=../models/$target_dir
function gen_mlir()
{
    onnx_path=../models/onnx/ram.onnx
    model_transform.py \
        --model_name $model_name \
        --model_def $onnx_path \
        --input_shapes [[$1,3,384,384]] \
        --pixel_format rgb  \
        --mean 123.675,116.28,103.53 \
        --scale 0.017125,0.017507,0.017429 \
        --mlir ${model_name}_$1b.mlir
}

function gen_cali_table()
{
    run_calibration.py ${model_name}_$1b.mlir \
    --dataset ../datasets/cali_set \
    --input_num 100 \
    --cali_method use_mse \
    -o ${model_name}_cali_table
}

function gen_int8bmodel()
{
    model_deploy.py \
        --mlir ${model_name}_$1b.mlir \
        --quantize INT8 \
        --chip $target \
        --calibration_table ${model_name}_cali_table \
        --quantize_table ${model_name}_qtable \
        --model ${model_name}_int8_$1b.bmodel
    mv ${model_name}_int8_$1b.bmodel $outdir/
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
model_name=ram
gen_mlir 1
gen_cali_table 1
gen_int8bmodel 1
popd