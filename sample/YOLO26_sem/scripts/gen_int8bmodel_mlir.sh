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
        --model_name yolo26s \
        --model_def ../models/onnx/yolo26s-sem.onnx \
        --input_shapes [[$1,3,1024,2048]] \
        --mean 0.0,0.0,0.0 \
        --scale 0.0039216,0.0039216,0.0039216 \
        --keep_aspect_ratio \
        --pixel_format rgb  \
        --mlir yolo26s_$1b.mlir
}

function gen_cali_table()
{
    # 语义分割量化校准：使用 Cityscapes 标定图片（RGB, 直接送入模型同款预处理）
    run_calibration.py yolo26s_$1b.mlir \
        --dataset ../datasets/cali \
        --input_num 12 \
        --part_asymmetric \
        --cali_method percentile9999 \
        --fp_type F32 \
        -o yolo26s_cali_table
}

function gen_int8bmodel()
{
    model_deploy.py \
        --mlir yolo26s_$1b.mlir \
        --quantize INT8 \
        --chip $target \
        --calibration_table yolo26s_cali_table \
        --model yolo26s_int8_$1b.bmodel

    mv yolo26s_int8_$1b.bmodel $outdir/
    if test $target = "bm1688";then
        model_deploy.py \
            --mlir yolo26s_$1b.mlir \
            --quantize INT8 \
            --chip $target \
            --calibration_table yolo26s_cali_table \
            --num_core 2 \
            --model yolo26s_int8_$1b_2core.bmodel

        mv yolo26s_int8_$1b_2core.bmodel $outdir/
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