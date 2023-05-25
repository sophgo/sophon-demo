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
        --model_name centernet \
        --model_def ../models/onnx/centernet_$1b.onnx \
        --input_shapes [[$1,3,512,512]] \
        --mean 104.01195,114.03422,119.91659 \
        --scale 0.014,0.014,0.014 \
        --keep_aspect_ratio \
        --pixel_format rgb  \
        --test_input ../datasets/ctdet_test.jpg \
        --test_result centernet_top_outputs.npz \
        --mlir centernet_$1b.mlir
}

function gen_cali_table()
{
    run_calibration.py centernet_$1b.mlir \
        --dataset ../datasets/coco128 \
        --input_num 200 \
        -o dlav0_cali_table
}

function gen_int8bmodel()
{
    model_deploy.py \
        --mlir centernet_$1b.mlir \
        --quantize INT8 \
        --chip $target \
        --test_input centernet_in_f32.npz \
        --test_reference centernet_top_outputs.npz \
        --calibration_table dlav0_cali_table \
        --quantize_table ../models/onnx/dlav0_qtable \
        --tolerance 0.8,0.5 \
        --model centernet_int8_$1b.bmodel 

    mv centernet_int8_$1b.bmodel $outdir/
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