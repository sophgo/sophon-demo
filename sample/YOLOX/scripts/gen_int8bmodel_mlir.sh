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
        --model_name yolox_s \
        --model_def ../models/onnx/yolox_s.onnx \
        --input_shapes [[$1,3,640,640]] \
        --mean 0.0,0.0,0.0 \
        --keep_aspect_ratio \
        --pixel_format rgb  \
        --mlir yolox_s_$1b.mlir
}

function gen_cali_table()
{
    run_calibration.py yolox_s_$1b.mlir \
        --dataset ../datasets/coco128/ \
        --input_num 128 \
        -o yolox_s_cali_table
}

function gen_int8bmodel()
{
    if [target="bm1684x"];then
        model_deploy.py \
            --mlir yolox_s_$1b.mlir \
            --quantize INT8 \
            --chip ${target} \
            --quantize_table ../models/onnx/yolox_s_qtable \
            --calibration_table yolox_s_cali_table \
            --model yolox_s_int8_$1b.bmodel
    elif [target="bm1684"];then
        model_deploy.py \
            --mlir yolox_s_$1b.mlir \
            --quantize INT8 \
            --chip ${target} \
            --calibration_table yolox_s_cali_table \
            --model yolox_s_int8_$1b.bmodel
    else 
        echo "not support chip."
    fi
    mv yolox_s_int8_$1b.bmodel $outdir/
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
