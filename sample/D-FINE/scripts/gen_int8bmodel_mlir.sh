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
    if [ ${model_name} = "dfine_n_coco" ]; then
        onnx_path="../models/onnx/dfine_n_coco.onnx"
    elif [ ${model_name} = "dfine_s_obj2coco" ]; then
        onnx_path="../models/onnx/dfine_s_obj2coco.onnx"
    fi
    model_transform.py \
        --model_name ${model_name} \
        --model_def $onnx_path \
        --input_shapes [[$1,3,640,640],[$1,2]] \
        --mean 0.0,0.0,0.0 \
        --scale 0.0039216,0.0039216,0.0039216 \
        --mlir ${model_name}_$1b.mlir
}

function gen_cali_table()
{
    run_calibration.py ${model_name}_$1b.mlir \
        --input_num 128 \
        --cali_method kl \
        --dataset ../datasets/cali_npz \
        -o ${model_name}_cali_table
}

function gen_int8bmodel()
{
    model_deploy.py \
        --mlir ${model_name}_$1b.mlir \
        --quantize INT8 \
        --chip $target \
        --calibration_table ${model_name}_cali_table \
        --quantize_table ${model_name}_qtable \
        --model ${model_name}_int8_$1b.bmodel

    mv ${model_name}_int8_$1b.bmodel $outdir/
    if test $target = "bm1688";then
        model_deploy.py \
            --mlir ${model_name}_$1b.mlir \
            --quantize INT8 \
            --chip $target \
            --calibration_table ${model_name}_cali_table \
            --quantize_table ${model_name}_qtable \
            --model ${model_name}_int8_$1b_2core.bmodel \
            --num_core 2

        mv ${model_name}_int8_$1b_2core.bmodel $outdir/
    fi
}


pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
model_name=${2:-dfine_n_coco}
gen_mlir 1
gen_cali_table 1
gen_int8bmodel 1

# batch_size=4
# gen_mlir 4
# gen_cali_table 4
# gen_int8bmodel 4

popd
