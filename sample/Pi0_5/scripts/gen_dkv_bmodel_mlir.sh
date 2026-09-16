#!/bin/bash
# gen_dkv_bmodel_mlir.sh — 编译 π0.5 主干（dual-path KV 前向）两段子模型
#
# 用法: ./gen_dkv_bmodel_mlir.sh [target]
#   target: bm1684x（默认）
#
# 输入: ../models/onnx/pi05_dkv0_9_fx.onnx    （18 层主干中的第 0–8 层）
#       ../models/onnx/pi05_dkv9_18_fx.onnx   （第 9–17 层）
# 输出: ../models/BM1684X/pi05_dkv0_9_w8bf16_1b.bmodel
#       ../models/BM1684X/pi05_dkv9_18_w8bf16_1b.bmodel
#
# 为什么拆两段: 18 层单图在 W8BF16 下编译会 OOM / 编译器崩溃；按 9 层切分后两段体积
#   各约 1 GB，可正常编译与加载。切分点是编译期约束，不是模型结构要求。
#
# 输入张量（4 个）:
#   prefix_embs [1,536,2048]   —— 视觉特征 + 语言 token 拼接后的前缀嵌入
#   p_amask     [1,1,536,536]  —— 前缀注意力掩码（含被屏蔽的死 token，值为 -10000）
#   p_cos       [1,536,256]    —— 前缀 RoPE cos
#   p_sin       [1,536,256]    —— 前缀 RoPE sin
# 输出张量: 每层一对 KV（共 18 对）+（仅第 0–8 段）hidden
#
# PL=536 的来历: 完整前缀为 968 token，其中 436 个被 p_amask 整段屏蔽（第 3 路补零黑图
#   256 个 + 语言 padding 180 个），对输出无影响，删除后序列长度降到 536。

model_dir=$(dirname $(readlink -f "$0"))

if [ ! $1 ]; then
    target=bm1684x
    target_dir=BM1684X
else
    target=${1,,}
    target_dir=${target^^}
fi

outdir=../models/$target_dir

PL=536
# dkv0_9 输出 18 个 KV + hidden；dkv9_18 只输出 18 个 KV
OUT_0_9=""
OUT_9_18=""
for i in $(seq 0 8); do
    OUT_0_9="$OUT_0_9,p_k$i,p_v$i"
done
OUT_0_9="${OUT_0_9#?},p_hidden"
for i in $(seq 9 17); do
    OUT_9_18="$OUT_9_18,p_k$i,p_v$i"
done
OUT_9_18="${OUT_9_18#?}"

gen_mlir()
{
    # $1 = 模型名（pi05_dkv0_9 / pi05_dkv9_18）, $2 = 输出张量列表
    model_transform.py \
        --model_name $1 \
        --model_def ../models/onnx/${1}_fx.onnx \
        --input_shapes "[[1,$PL,2048],[1,1,$PL,$PL],[1,$PL,256],[1,$PL,256]]" \
        --output_names "$2" \
        --mlir $1.mlir
}

gen_w8bf16bmodel()
{
    model_deploy.py \
        --mlir $1.mlir \
        --quantize W8BF16 \
        --chip $target \
        --model ${1}_w8bf16_1b.bmodel
    mv ${1}_w8bf16_1b.bmodel $outdir/
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
gen_mlir pi05_dkv0_9  "$OUT_0_9"
gen_mlir pi05_dkv9_18 "$OUT_9_18"
gen_w8bf16bmodel pi05_dkv0_9
gen_w8bf16bmodel pi05_dkv9_18
popd
