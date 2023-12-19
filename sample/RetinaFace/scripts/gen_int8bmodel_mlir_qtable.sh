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
        --model_name retinaface_mobilenet0.25 \
        --model_def ../data/models/onnx/retinaface_mobilenet0.25.onnx \
        --input_shapes [[$1,3,640,640]] \
        --mean 104.0,117.0,123.0 \
        --scale 1.0,1.0,1.0 \
        --keep_aspect_ratio \
        --pixel_format bgr  \
        --mlir retinaface_mobilenet0.25_$1b.mlir
}

function gen_cali_table()
{
    run_calibration.py retinaface_mobilenet0.25_$1b.mlir \
        --dataset ../data/images/WIDERVAL \
        --input_num 100 \
        -o retinaface_mobilenet_cali_table
}

function gen_qtable()
{
    run_qtable.py retinaface_mobilenet0.25_$1b.mlir \
        --dataset ../data/images/WIDERVAL \
        --input_num 100 \
        --expected_cos 0.99 \
        --calibration_table retinaface_mobilenet_cali_table \
        --chip $target \
        -o retinaface_mobilenet0.25_qtable

    # only keep top3 fp32 layers
    head -9 retinaface_mobilenet0.25_qtable > retinaface_mobilenet0.25_qtable_keep_top3
}

function gen_int8bmodel()
{
    model_deploy.py \
        --mlir retinaface_mobilenet0.25_$1b.mlir \
        --quantize INT8 \
        --chip $target \
        --calibration_table retinaface_mobilenet_cali_table \
        --quantize_table retinaface_mobilenet0.25_qtable_keep_top3 \
        --model retinaface_mobilenet0.25_int8_$1b.bmodel

    mv retinaface_mobilenet0.25_int8_$1b.bmodel $outdir/

    if test $target = "bm1688";then
        model_deploy.py \
            --mlir retinaface_mobilenet0.25_$1b.mlir \
            --quantize INT8 \
            --chip $target \
            --model retinaface_mobilenet0.25_int8_$1b_2core.bmodel \
            --calibration_table retinaface_mobilenet_cali_table \
            --quantize_table retinaface_mobilenet0.25_qtable_keep_top3 \
            --num_core 2 \
            --debug 
        mv retinaface_mobilenet0.25_int8_$1b_2core.bmodel $outdir/
    fi
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
batch_size=1
gen_mlir 1
gen_cali_table 1
gen_qtable 1
gen_int8bmodel 1

# batch_size=4
gen_mlir 4
gen_int8bmodel 4

popd