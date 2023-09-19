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
        --model_name yolov5s_v6.1_3output \
        --model_def ../models/onnx/yolov5s_v6.1_3output.onnx \
        --input_shapes [[$1,3,640,640]] \
        --mean 0.0,0.0,0.0 \
        --scale 0.0039216,0.0039216,0.0039216 \
        --keep_aspect_ratio \
        --pixel_format rgb  \
        --test_input ../datasets/test/3.jpg \
        --test_result yolov5_top.npz \
        --mlir yolov5s_v6.1_3output_$1b.mlir
}

function gen_cali_table()
{
    run_calibration.py yolov5s_v6.1_3output_$1b.mlir \
        --dataset ../datasets/coco128/ \
        --input_num 64 \
        -o yolov5s_cali_table
}

function gen_int8bmodel()
{
    model_deploy.py \
        --mlir yolov5s_v6.1_3output_$1b.mlir \
        --quantize INT8 \
        --chip $target \
        --calibration_table yolov5s_cali_table \
        --model yolov5s_v6.1_3output_int8_$1b.bmodel \
        --test_input ../datasets/test/3.jpg \
        --test_reference yolov5_top.npz \
        --debug 

    mv yolov5s_v6.1_3output_int8_$1b.bmodel $outdir/
    if test $target = "bm1688";then
        model_deploy.py \
            --mlir yolov5s_v6.1_3output_$1b.mlir \
            --quantize INT8 \
            --chip $target \
            --model yolov5s_v6.1_3output_int8_$1b_2core.bmodel \
            --calibration_table yolov5s_cali_table \
            --num_core 2 \
            --test_input ../datasets/test/3.jpg \
            --test_reference yolov5_top.npz \
            --debug 
        mv yolov5s_v6.1_3output_int8_$1b_2core.bmodel $outdir/
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

# batch_size=4
gen_mlir 4
gen_int8bmodel 4

popd