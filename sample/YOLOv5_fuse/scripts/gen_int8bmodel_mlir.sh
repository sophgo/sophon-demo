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
        --chip  $target \
        --processor  $target \
        --fuse_preprocess \
        --calibration_table yolov5s_cali_table \
        --customization_format BGR_PACKED \
        --model yolov5s_v6.1_3output_int8_$1b.bmodel

    mv yolov5s_v6.1_3output_int8_$1b.bmodel $outdir/
    if test $target = "bm1688";then
        model_deploy.py \
            --mlir yolov5s_v6.1_3output_$1b.mlir \
            --quantize INT8 \
            --chip  $target \
            --processor  $target \
            --fuse_preprocess \
            --calibration_table yolov5s_cali_table \
            --customization_format BGR_PACKED \
            --num_core 2 \
            --model yolov5s_v6.1_3output_int8_$1b_2core.bmodel

        mv yolov5s_v6.1_3output_int8_$1b_2core.bmodel $outdir/
    fi
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
gen_cali_table 1
gen_int8bmodel 1

# batch_size=4
gen_int8bmodel 4

popd