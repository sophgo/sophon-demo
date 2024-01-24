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
fp_forward_chip=$target
if test $target = "bm1688"; then
    fp_forward_chip=bm1684x
fi

function gen_mlir()
{
   model_transform.py \
        --model_name yolox_s \
        --model_def ../models/onnx/yolox_s.onnx \
        --input_shapes [[$1,3,640,640]] \
        --mean 0.0,0.0,0.0 \
        --keep_aspect_ratio \
        --pixel_format bgr  \
        --test_input ../datasets/test/3.jpg \
        --test_result yolox_top.npz \
        --mlir yolox_s_$1b.mlir
}

function gen_cali_table()
{
    run_calibration.py yolox_s_$1b.mlir \
        --dataset ../datasets/coco128/ \
        --input_num 128 \
        --debug_cmd "MAX" \
        -o yolox_s_cali_table
}

function gen_qtable()
{
    fp_forward.py yolox_s_$1b.mlir \
        --quantize INT8 \
        --chip $fp_forward_chip \
        --fpfwd_inputs /backbone/backbone/dark2/dark2.1/conv3/conv/Conv_output_0_Conv \
        --fpfwd_outputs /head/stems.0/conv/Conv_output_0_Conv,/head/stems.1/conv/Conv_output_0_Conv,/head/stems.2/conv/Conv_output_0_Conv \
        -o yolox_s_qtable

}


function gen_int8bmodel()
{
    model_deploy.py \
        --mlir yolox_s_$1b.mlir \
        --quantize INT8 \
        --chip ${target} \
        --quantize_table yolox_s_qtable \
        --calibration_table yolox_s_cali_table \
        --test_input ../datasets/test/3.jpg \
        --test_reference yolox_top.npz \
        --debug \
        --model yolox_s_int8_$1b.bmodel

    mv yolox_s_int8_$1b.bmodel $outdir/

    if test $target = "bm1688";then
        model_deploy.py \
            --mlir yolox_s_$1b.mlir \
            --quantize INT8 \
            --chip ${target} \
            --quantize_table yolox_s_qtable \
            --calibration_table yolox_s_cali_table \
            --test_input ../datasets/test/3.jpg \
            --test_reference yolox_top.npz \
            --num_core 2 \
            --debug \
            --model yolox_s_int8_$1b_2core.bmodel

        mv yolox_s_int8_$1b_2core.bmodel $outdir/
    fi
}


pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
gen_mlir 1
gen_cali_table 1
gen_qtable 1
gen_int8bmodel 1

# batch_size=4
gen_mlir 4
gen_int8bmodel 4
