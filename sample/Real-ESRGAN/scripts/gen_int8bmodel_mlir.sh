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
        --model_name real_esrgan \
        --model_def ../models/onnx/realesr-general-x4v3.onnx \
        --input_shapes [[$1,3,480,640]] \
        --mean 0.0,0.0,0.0 \
        --scale 0.0039216,0.0039216,0.0039216 \
        --keep_aspect_ratio \
        --pixel_format rgb  \
        --test_input ../datasets/coco128/000000000127.jpg \
        --test_result real_esrgan_top.npz \
        --mlir real_esrgan_$1b.mlir
}

function gen_cali_table()
{
    run_calibration.py real_esrgan_$1b.mlir \
        --dataset ../datasets/coco128/ \
        --input_num 16 \
        -o real_esrgan_cali_table
}

function gen_int8bmodel()
{
    model_deploy.py \
        --mlir real_esrgan_$1b.mlir \
        --quantize INT8 \
        --chip $target \
        --calibration_table real_esrgan_cali_table \
        --model real_esrgan_int8_$1b.bmodel \
        --test_input ../datasets/coco128/000000000127.jpg \
        --test_reference real_esrgan_top.npz \
        --quant_input \
        --quant_output \
        --compare_all

    mv real_esrgan_int8_$1b.bmodel $outdir/
    if test $target = "bm1688";then
        model_deploy.py \
            --mlir real_esrgan_$1b.mlir \
            --quantize INT8 \
            --chip $target \
            --model real_esrgan_int8_$1b_2core.bmodel \
            --calibration_table real_esrgan_cali_table \
            --num_core 2 \
            --test_input ../datasets/coco128/000000000127.jpg \
            --test_reference real_esrgan_top.npz \
            --quant_input \
            --quant_output
        mv real_esrgan_int8_$1b_2core.bmodel $outdir/
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