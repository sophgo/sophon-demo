#!/bin/bash
# gen_siglip_bmodel_mlir.sh — 编译 π0.5 视觉编码器（SigLIP So400m/14）子模型
#
# 用法: ./gen_siglip_bmodel_mlir.sh [target]
#   target: bm1684x（默认）
#
# 输入: ../models/onnx/pi05_siglip.onnx（siglip 不做图修补，直接用导出产物）
# 输出: ../models/BM1684X/pi05_siglip_w8bf16_2b.bmodel
#
# 注意: batch 维固定为 2 —— π0.5 每帧需要两张图（主视角 agentview + 腕部 wrist），
#       合成一个 batch 只加载一遍权重。这是本 sample 的关键优化之一，不是"2 个样本"。

model_dir=$(dirname $(readlink -f "$0"))

if [ ! $1 ]; then
    target=bm1684x
    target_dir=BM1684X
else
    target=${1,,}
    target_dir=${target^^}
fi

outdir=../models/$target_dir

function gen_mlir_siglip()
{
    model_transform.py \
        --model_name pi05_siglip \
        --model_def ../models/onnx/pi05_siglip.onnx \
        --input_shapes "[[2,3,224,224]]" \
        --mlir pi05_siglip.mlir
}

function gen_w8bf16bmodel_siglip()
{
    model_deploy.py \
        --mlir pi05_siglip.mlir \
        --quantize W8BF16 \
        --chip $target \
        --model pi05_siglip_w8bf16_2b.bmodel
    mv pi05_siglip_w8bf16_2b.bmodel $outdir/
}

# 如需 F16 对照档：把上面 model_deploy 的 --quantize 改成 F16、输出名改成 pi05_siglip_f16_2b.bmodel

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
gen_mlir_siglip
gen_w8bf16bmodel_siglip
# gen_f16bmodel_siglip   # 对照档：精度更高但体积 ×2，默认不编译
popd
