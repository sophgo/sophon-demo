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

# DINOv2 官方 eval 口径：输入 224x224（Resize 256 BICUBIC + CenterCrop 224），
# 归一化 mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]（RGB 序）。
# 这里的 --mean/--scale 按「模型输入通道序（RGB）」给出，与推理脚本中 bmcv
# convert_to 的 (a,b) = (1/(255*std), -mean/std) 完全一致。
function gen_mlir()
{
    model_transform.py \
        --model_name dinov2_vitb14_reg_$1b \
        --model_def ../models/onnx/dinov2_vitb14_reg_224_$1b.onnx \
        --input_shapes [[$1,3,224,224]] \
        --mean 123.675,116.28,103.53 \
        --scale 0.01712475,0.017507,0.01742919 \
        --pixel_format rgb \
        --test_input ../datasets/test_input/dinov2_$1b_in_f32.npz \
        --test_result dinov2_vitb14_reg_$1b_top_outputs.npz \
        --mlir dinov2_vitb14_reg_$1b.mlir
}

function gen_fp32bmodel()
{
    model_deploy.py \
        --mlir dinov2_vitb14_reg_$1b.mlir \
        --quantize F32 \
        --chip $target \
        --model dinov2_vitb14_reg_fp32_$1b.bmodel \
        --test_input dinov2_vitb14_reg_$1b_in_f32.npz \
        --test_reference dinov2_vitb14_reg_$1b_top_outputs.npz

    mv dinov2_vitb14_reg_fp32_$1b.bmodel $outdir/
    if test $target = "bm1688";then
        model_deploy.py \
            --mlir dinov2_vitb14_reg_$1b.mlir \
            --quantize F32 \
            --chip $target \
            --model dinov2_vitb14_reg_fp32_$1b_2core.bmodel \
            --num_core 2 \
            --test_input dinov2_vitb14_reg_$1b_in_f32.npz \
            --test_reference dinov2_vitb14_reg_$1b_top_outputs.npz
        mv dinov2_vitb14_reg_fp32_$1b_2core.bmodel $outdir/
    fi
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
gen_mlir 1
gen_fp32bmodel 1

popd
