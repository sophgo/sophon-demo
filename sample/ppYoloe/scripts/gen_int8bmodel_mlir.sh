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
        --model_name ppyoloe \
        --model_def ../models/onnx/ppyoloe.onnx \
        --input_shapes [[$1,3,640,640],[$1,2]] \
        --mlir ppyoloe_$1b.mlir \
        --output_names p2o.Concat.29,p2o.Div.1
}

function gen_cali_table()
{
    run_calibration.py ppyoloe_$1b.mlir \
        --dataset ../datasets/coco128_npz/ \
        --input_num 128 \
        --cali_method use_percentile9999 \
        -o ppyoloe_cali_table
}

function gen_int8bmodel()
{
    model_deploy.py \
        --mlir ppyoloe_$1b.mlir \
        --quantize INT8 \
        --chip $target \
        --calibration_table ppyoloe_cali_table \
        --quantize_table ppyoloe_qtable \
        --model ppyoloe_int8_$1b.bmodel
        # --test_input ../datasets/test/3.jpg \
        # --test_reference ppyoloe_top.npz

    mv ppyoloe_int8_$1b.bmodel $outdir/
    if test $target = "bm1688";then
        model_deploy.py \
            --mlir ppyoloe_$1b.mlir \
            --quantize INT8 \
            --chip $target \
            --model ppyoloe_int8_$1b_2core.bmodel \
            --calibration_table ppyoloe_cali_table \
            --quantize_table ppyoloe_qtable \
            --num_core 2
            # --test_input ../datasets/test/3.jpg \
            # --test_reference ppyoloe_top.npz \
        mv ppyoloe_int8_$1b_2core.bmodel $outdir/
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