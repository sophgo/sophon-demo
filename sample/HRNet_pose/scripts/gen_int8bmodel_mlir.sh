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
        --model_name hrnetint8 \
        --model_def ../models/onnx/pose_hrnet_w32_256x192.onnx \
        --input_shapes [[1,3,256,192]] \
        --mean 123.675,123.675,103.53 \
        --scale 0.01712475,0.017507,0.01742919 \
        --keep_aspect_ratio \
        --pixel_format rgb \
        --test_input ../mlir_utils/person.jpg \
        --test_result hrnet_top_outputs.npz \
        --mlir hrnetint8.mlir
}

function gen_cali_table()
{
    run_calibration.py hrnetint8.mlir \
        --dataset ../datasets/single_person_images_100/ \
        --input_num 64 \
        -o hrnet_cali_table
}

function gen_int8bmodel()
{
    model_deploy.py \
        --mlir hrnetint8.mlir \
        --quantize INT8 \
        --chip $target \
        --calibration_table hrnet_cali_table \
        --test_input hrnetint8_in_f32.npz \
        --test_reference hrnet_top_outputs.npz \
        --quantize_table ../mlir_utils/qtable \
        --model hrnet_w32_256x192_int8.bmodel

    mv hrnet_w32_256x192_int8.bmodel $outdir/
    if test $target = "bm1688";then
        model_deploy.py \
            --mlir hrnetint8.mlir \
            --quantize INT8 \
            --chip $target \
            --calibration_table hrnet_cali_table \
            --num_core 2 \
            --test_input hrnetint8_in_f32.npz \
            --test_reference hrnet_top_outputs.npz \
            --quantize_table ../mlir_utils/qtable \
            --model hrnet_w32_256x192_int8_2core.bmodel 

        mv hrnet_w32_256x192_int8_2core.bmodel $outdir/
    fi
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
gen_mlir
gen_cali_table
gen_int8bmodel

popd