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

function gen_mlir_int8()
{
    model_transform.py \
        --model_name mobilenetv4_conv_medium_$1b_int8 \
        --model_def ../models/torch/mobilenetv4_conv_medium.torchscript.pt \
        --input_shapes [[$1,3,224,224]] \
        --mean 103.53,116.28,123.67 \
        --scale 0.01742919,0.017507,0.01712475 \
        --pixel_format rgb  \
        --test_input ../datasets/cali_data/ILSVRC2012_val_00000555.jpg \
        --test_result mobilenetv4_conv_medium_$1b_top_outputs_int8.npz \
        --mlir mobilenetv4_conv_medium_$1b_int8.mlir
    cp -r mobilenetv4_conv_medium_$1b_int8.mlir onnx.mlir
}

function gen_cali_table()
{
    run_calibration.py mobilenetv4_conv_medium_$1b_int8.mlir \
        --dataset ../datasets/cali_data \
        --input_num 200 \
        -o mobilenetv4_conv_medium_cali_table
}

function gen_int8bmodel()
{
    model_deploy.py \
        --mlir mobilenetv4_conv_medium_$1b_int8.mlir \
        --quantize INT8 \
        --chip $target \
        --calibration_table mobilenetv4_conv_medium_cali_table \
        --model mobilenetv4_conv_medium_int8_$1b.bmodel \
        --test_input mobilenetv4_conv_medium_$1b_int8_in_f32.npz \
        --test_reference mobilenetv4_conv_medium_$1b_top_outputs_int8.npz

    mv mobilenetv4_conv_medium_int8_$1b.bmodel $outdir/
    if test $target = "bm1688";then
        model_deploy.py \
            --mlir mobilenetv4_conv_medium_$1b_int8.mlir \
            --quantize INT8 \
            --chip $target \
            --model mobilenetv4_conv_medium_int8_$1b_2core.bmodel \
            --calibration_table mobilenetv4_conv_medium_cali_table \
            --test_input mobilenetv4_conv_medium_$1b_int8_in_f32.npz \
            --test_reference mobilenetv4_conv_medium_$1b_top_outputs_int8.npz \
            --num_core 2
        mv mobilenetv4_conv_medium_int8_$1b_2core.bmodel $outdir/
    fi
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
gen_mlir_int8 1
gen_cali_table 1
gen_int8bmodel 1

# batch_size=4
gen_mlir_int8 4
gen_int8bmodel 4

popd
