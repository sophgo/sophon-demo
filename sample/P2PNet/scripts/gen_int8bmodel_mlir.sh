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

gen_mlir()
{
    model_transform.py \
        --model_name p2pnet \
        --model_def ../models/onnx/p2pnet_$1b.onnx \
        --input_shapes [[$1,3,512,512]] \
        --mean 123.675,116.28,103.53 \
        --scale 0.01712,0.01751,0.01743 \
        --keep_aspect_ratio \
        --pixel_format rgb \
        --test_input ../datasets/calibration/IMG_1.jpg \
        --test_result p2pnet_$1b_top_outputs.npz \
        --mlir p2pnet_$1b.mlir
}
gen_cali_table()
{
    run_calibration.py p2pnet_$1b.mlir \
        --dataset ../datasets/calibration \
        --input_num 300 \
        -o p2pnet_cali_table
}

gen_int8bmodel()
{
    model_deploy.py \
        --mlir p2pnet_$1b.mlir \
        --quantize INT8 \
		--calibration_table p2pnet_cali_table \
        --chip ${target} \
        --model p2pnet_${target}_int8_$1b.bmodel \
        --test_input ../datasets/calibration/IMG_1.jpg \
        --test_reference p2pnet_$1b_top_outputs.npz

    mv p2pnet_${target}_int8_$1b.bmodel $outdir/
    if test $target = "bm1688";then
        model_deploy.py \
            --mlir p2pnet_$1b.mlir \
            --quantize INT8 \
            --chip $target \
            --model p2pnet_int8_$1b_2core.bmodel \
            --calibration_table p2pnet_cali_table \
            --test_input ../datasets/calibration/IMG_1.jpg \
            --test_reference p2pnet_$1b_top_outputs.npz \
            --num_core 2
        mv p2pnet_int8_$1b_2core.bmodel $outdir/
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