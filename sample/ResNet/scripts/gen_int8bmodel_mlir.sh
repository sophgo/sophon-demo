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
    # use pt to cali, can achieve higher acc than onnx.
    model_transform.py \
        --model_name resnet50_$1b \
        --model_def ../models/torch/resnet50-11ad3fa6.torchscript.pt \
        --input_shapes [[$1,3,224,224]] \
        --mean 103.53,116.28,123.67 \
        --scale 0.01742919,0.017507,0.01712475 \
        --pixel_format rgb  \
        --test_input ../datasets/cali_data/ILSVRC2012_val_00000555.jpg \
        --test_result resnet50_$1b_top_outputs.npz \
        --mlir resnet50_$1b.mlir
}

function gen_cali_table()
{
    run_calibration.py resnet50_$1b.mlir \
        --dataset ../datasets/cali_data \
        --input_num 200 \
        -o resnet50_cali_table
    # run_qtable.py resnet50_$1b.mlir \
    #     --dataset ../datasets/cali_data \
    #     --input_num 25 \
    #     --calibration_table resnet50_cali_table \
    #     --chip $target \
    #     -o resnet50_qtable
}

function gen_int8bmodel()
{
    model_deploy.py \
        --mlir resnet50_$1b.mlir \
        --quantize INT8 \
        --chip $target \
        --calibration_table resnet50_cali_table \
        --test_input resnet50_$1b_in_f32.npz \
        --test_reference resnet50_$1b_top_outputs.npz \
        --model resnet50_int8_$1b.bmodel
        #--debug

    mv resnet50_int8_$1b.bmodel $outdir/
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
