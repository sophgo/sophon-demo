#!/bin/bash
# 生成 FP16 bmodel (MoGe-2 ViT-L/14, 静态输入 1x3x518x518, fp16 权重)
# 用法: ./scripts/gen_fp16bmodel_mlir.sh [target]    target: bm1684x/bm1688/cv186x
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
    model_transform.py \
        --model_name $model_name \
        --model_def $onnx_path \
        --input_shapes [[$1,3,518,518]] \
        --pixel_format rgb \
        --mlir ${model_name}_$1b.mlir
}

function gen_fp16bmodel()
{
    model_deploy.py \
        --mlir ${model_name}_$1b.mlir \
        --quantize F16 \
        --chip $target \
        --model ${model_name}_fp16_$1b.bmodel

    mv ${model_name}_fp16_$1b.bmodel $outdir/
    if test $target = "bm1688"; then
        model_deploy.py \
            --mlir ${model_name}_$1b.mlir \
            --quantize F16 \
            --chip $target \
            --model ${model_name}_fp16_$1b_2core.bmodel \
            --num_core 2
        mv ${model_name}_fp16_$1b_2core.bmodel $outdir/
    fi
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1 (静态形状固定为 1)
gen_mlir 1
gen_fp16bmodel 1
popd