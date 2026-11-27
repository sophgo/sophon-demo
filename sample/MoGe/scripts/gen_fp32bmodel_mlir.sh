#!/bin/bash
# 生成 FP32 bmodel (MoGe-2 ViT-L/14, 静态输入 1x3x518x518, fp32)
# 用法: ./scripts/gen_fp32bmodel_mlir.sh [target]    target: bm1684x/bm1688/cv186x
model_dir=$(dirname $(readlink -f "$0"))

if [ ! $1 ]; then
    target=bm1688
    target_dir=BM1688
else
    target=${1,,}
    target_dir=${target^^}
fi

outdir=../models/$target_dir
model_name=moge-2-vitl-normal
onnx_path=../models/onnx/$model_name.onnx

function gen_mlir()
{
    # MoGe 的 ImageNet 均值/方差已内置于 DINOv2 encoder 中，输入即 RGB float [0,1]，
    # 因此 model_transform 不做归一化(mean=0, scale=1)，与官方 ONNX 导出保持一致。
    model_transform.py \
        --model_name $model_name \
        --model_def $onnx_path \
        --input_shapes [[$1,3,518,518]] \
        --pixel_format rgb \
        --mlir ${model_name}_$1b.mlir
}

function gen_fp32bmodel()
{
    model_deploy.py \
        --mlir ${model_name}_$1b.mlir \
        --quantize F32 \
        --chip $target \
        --model ${model_name}_fp32_$1b.bmodel

    mv ${model_name}_fp32_$1b.bmodel $outdir/
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1 (静态形状固定为 1)
gen_mlir 1
gen_fp32bmodel 1
popd