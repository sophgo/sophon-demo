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
        --model_name yolov10s \
        --model_def ../models/onnx/yolov10s.onnx \
        --input_shapes [[$1,3,640,640]] \
        --mean 0.0,0.0,0.0 \
        --scale 0.0039216,0.0039216,0.0039216 \
        --keep_aspect_ratio \
        --pixel_format rgb  \
        --mlir yolov10s_$1b.mlir
}

function gen_cali_table()
{
    run_calibration.py yolov10s_$1b.mlir \
        --dataset ../datasets/coco128/ \
        --input_num 32 \
        -o yolov10s_cali_table
}

function gen_int8bmodel()
{
    qtable_path=../models/onnx/yolov10s_qtable_mix

    model_deploy.py \
        --mlir yolov10s_$1b.mlir \
        --quantize INT8 \
        --chip $target \
        --quantize_table $qtable_path \
        --calibration_table yolov10s_cali_table \
        --model yolov10s_int8_$1b.bmodel 

    mv yolov10s_int8_$1b.bmodel $outdir/
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