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

#$1 batch size
function gen_mlir_superpoint()
{
    model_transform.py \
        --model_name superpoint \
        --model_def ../models/onnx/superpoint_to_nms.onnx \
        --input_shapes [[$1,1,360,640]] \
        --mlir superpoint_$1b.mlir \
        --test_input ../datasets/superpoint_test_input/1160sat.npy \
        --test_result superpoint_test_result.npz
}

#$1 batch size, $2 max keypoint size
function gen_fp16bmodel_superpoint()
{
    model_deploy.py \
        --mlir superpoint_$1b.mlir \
        --quantize F16 \
        --chip $target \
        --model superpoint_fp16_$1b.bmodel \
        --quantize_table ../models/onnx/superpoint_fp16_qtable \
        --test_input superpoint_in_f32.npz \
        --test_reference superpoint_test_result.npz
    mv superpoint_fp16_$1b.bmodel $outdir/
    # if test $target = "bm1688";then
    #     model_deploy.py \
    #         --mlir superpoint_$1b.mlir \
    #         --quantize F16 \
    #         --chip $target \
    #         --model superpoint_fp16_$1b_2core.bmodel \
    #         --num_core 2 \
    #         --quantize_table ../models/onnx/superpoint_fp16_qtable \
    #         --test_input ../datasets/superpoint_test_input/1160sat.npy \
    #         --test_reference superpoint_test_result.npz
    #     mv superpoint_fp16_$1b_2core.bmodel $outdir/
    # fi
}

#$1 batch size, $2 max keypoint size
function gen_mlir_superglue()
{
    #this superglue onnx only for 360,640
    model_transform.py \
        --model_name superglue \
        --model_def ../models/onnx/superglue_indoor_iter$3_$2.onnx \
        --input_shapes [[$1,$2,2],[$1,$2],[$1,256,$2],[$1,$2,2],[$1,$2],[$1,256,$2]] \
        --mlir superglue_$1b_iter$3_$2.mlir
        # --test_input ../datasets/superglue_test_input/1160fli_1160sat.npz \
        # --test_result superglue_test_result.npz
        # --debug
}

#$1 batch size, $2 max keypoint size
function gen_fp16bmodel_superglue()
{
    model_deploy.py \
        --mlir superglue_$1b_iter$3_$2.mlir \
        --quantize F16 \
        --chip $target \
        --model superglue_fp16_$1b_iter$3_$2.bmodel
        # --quantize_table test_qtable \
        # --test_input ../datasets/superglue_test_input/1160fli_1160sat.npz \
        # --test_reference superglue_test_result.npz \
        # --compare_all \
        # --tolerance 0.99,0.99
        # --debug

    mv superglue_fp16_$1b_iter$3_$2.bmodel $outdir/
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
gen_mlir_superpoint 1
gen_fp16bmodel_superpoint 1

gen_mlir_superglue 1 1024 20
gen_fp16bmodel_superglue 1 1024 20
popd