#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))

if [ ! $1 ]; then
    target=bm1688
    target_dir=bm1688
else
    target=${1,,}
    target_dir=${target^^}
fi

outdir=../models/$target_dir

function gen_mlir_superglue()
{
    model_transform.py \
        --model_name superglue \
        --model_def ../models/onnx/superglue_indoor_iter$3_$2.onnx \
        --input_shapes [[$1,$2,2],[$1,$2],[$1,256,$2],[$1,$2,2],[$1,$2],[$1,256,$2]] \
        --test_input ../datasets/superglue_test_input/1.npz \
        --test_result superglue_test_result.npz \
        --mlir superglue_$1b_iter$3_$2.mlir
}

# 生成cali_table的步骤已省略

function gen_int8bmodel_superglue()
{
    model_deploy.py \
        --mlir superglue_$1b_iter$3_$2.mlir \
        --quantize INT8 \
        --chip $target \
        --calibration_table ../models/BM1688/int8/superglue_cali_table \  
        --model superglue_int8_$1b_iter$3_$2.bmodel \
        --quantize_table ../models/BM1688/int8/qtable_superglue

    mv superglue_int8_$1b_iter$3_$2.bmodel $outdir/
    if test $target = "bm1688";then
        model_deploy.py \
            --mlir superglue_$1b_iter$3_$2.mlir \
            --quantize INT8 \
            --chip $target \
            --model superglue_int8_$1b_2core.bmodel \
            --calibration_table ../models/BM1688/int8/superglue_cali_table \
            --quantize_table ../models/BM1688/int8/qtable_superglue \
            --num_core 2 
        mv superglue_int8_$1b_2core.bmodel $outdir/
    fi
}


function gen_mlir_superpoint()
{
    model_transform.py \
        --model_name superpoint \
        --model_def ../models/onnx/superpoint_to_nms.onnx \
        --input_shapes [[$1,1,360,640]] \
        --test_input ../datasets/superpoint_test_input/1160sat.npy \
        --test_result superpoint_test_result.npz \
        --mlir superpoint_$1b.mlir 
}

# 生成cali_table的步骤已省略

function gen_int8bmodel_superpoint()
{
    model_deploy.py \
        --mlir superpoint_$1b.mlir \
        --quantize INT8 \
        --chip bm1688 \
        --quantize_table  ../models/BM1688/int8/qtable_superpoint \
        --calibration_table ../models/BM1688/int8/superpoint_cali_table \
        --model superpoint_int8_$1b.bmodel \
        --test_input ../datasets/superpoint_test_input/1160sat.npy \
        --test_reference superpoint_test_result.npz \
        --tolerance 0.5,0 \
        --debug 

    mv superpoint_int8_$1b.bmodel $outdir/

    if test $target = "bm1688";then
        model_deploy.py \
            --mlir superpoint_$1b.mlir \
            --quantize INT8 \
            --chip bm1688 \
            --quantize_table ../models/BM1688/int8/qtable_superpoint \
            --calibration_table ../models/BM1688/int8/superpoint_cali_table \
            --model superpoint_int8_$1b_2core.bmodel \
            --test_input ../datasets/superpoint_test_input/1160sat.npy \
            --test_reference superpoint_test_result.npz \
            --tolerance 0.5,0 \
            --num_core 2 \
            --debug 
            
        mv superpoint_int8_$1b_2core.bmodel $outdir/
    fi
}


pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# # batch_size=1
gen_mlir_superglue 1 1024 20
gen_int8bmodel_superglue 1 1024 20

gen_mlir_superpoint 1
gen_int8bmodel_superpoint 1


popd
