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
        --model_name ppyolov3 \
        --model_def ../models/onnx/ppyolov3_$1b.onnx \
        --input_shapes [[$1,3,608,608]] \
        --output_names Transpose_0,Transpose_2,Transpose_4 \
        --mean 123.675,116.28,103.53 \
        --scale 0.01712475,0.017507,0.0174292 \
        --mlir ppyolov3_$1b.mlir
}
function gen_cali_table()
{
    run_calibration.py ppyolov3_$1b.mlir \
        --dataset ../datasets/coco128/ \
        --input_num 128 \
        -o ppyolov3_cali_table_$1b
}
function gen_int8bmodel()
{
    model_deploy.py \
        --mlir ppyolov3_$1b.mlir \
        --quantize INT8 \
        --chip $target \
        --calibration_table ppyolov3_cali_table_$1b \
        --model ppyolov3_int8_$1b.bmodel

    mv ppyolov3_int8_$1b.bmodel $outdir/
    
    if test $target = "bm1688";then
        model_deploy.py \
            --mlir ppyolov3_$1b.mlir \
            --quantize INT8 \
            --chip $target \
            --num_core 2 \
            --calibration_table ppyolov3_cali_table_$1b \
            --model ppyolov3_int8_$1b_2core.bmodel
    
        mv ppyolov3_int8_$1b_2core.bmodel $outdir/
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

popd