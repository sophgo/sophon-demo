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
        --model_name scrfd_10g_kps_$1b \
        --model_def ../models/onnx/scrfd_10g_kps_$1b.onnx \
        --input_shapes [[$1,3,640,640]] \
        --mean 127.5,127.5,127.5 \
        --scale 0.0078125,0.0078125,0.0078125 \
        --keep_aspect_ratio \
        --pixel_format rgb  \
        --test_input ../datasets/test/men.jpg \
        --test_result scrfd_top.npz \
        --mlir scrfd_10g_kps_$1b.mlir
}

function gen_cali_table()
{
    run_calibration.py scrfd_10g_kps_$1b.mlir \
        --dataset ../datasets/WIDER_val/0--Parade/ \
        --input_num 64 \
        -o scrfd_cali_table
}

function gen_int8bmodel()
{
    model_deploy.py \
        --mlir scrfd_10g_kps_$1b.mlir \
        --quantize INT8 \
        --chip $target \
        --calibration_table scrfd_cali_table \
        --model scrfd_10g_kps_int8_$1b.bmodel \
        --test_input ../datasets/test/men.jpg \
        --test_reference scrfd_top.npz \
        --debug 

    mv scrfd_10g_kps_int8_$1b.bmodel $outdir/
    if test $target = "bm1688";then
        model_deploy.py \
            --mlir scrfd_10g_kps_$1b.mlir \
            --quantize INT8 \
            --chip $target \
            --model scrfd_10g_kps_int8_$1b_2core.bmodel \
            --calibration_table scrfd_cali_table \
            --num_core 2 \
            --test_input ../datasets/test/men.jpg \
            --test_reference scrfd_top.npz \
            --debug 
        mv scrfd_10g_kps_int8_$1b_2core.bmodel $outdir/
    fi
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
gen_mlir 1
gen_cali_table 1
gen_int8bmodel 1

# # batch_size=4
gen_mlir 4
gen_cali_table 4
gen_int8bmodel 4

popd