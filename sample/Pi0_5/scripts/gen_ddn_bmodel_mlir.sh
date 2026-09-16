#!/bin/bash
# gen_ddn_bmodel_mlir.sh — 编译 π0.5 动作专家（flow-matching 去噪）三段子模型
#
# 用法: ./gen_ddn_bmodel_mlir.sh [target]
#   target: bm1684x（默认）
#
# 输入: ../models/onnx/pi05_ddn0_6_fx.onnx
#       ../models/onnx/pi05_ddn6_12_fx.onnx
#       ../models/onnx/pi05_ddn12_18_fx.onnx
# 输出: ../models/BM1684X/pi05_ddn0_6_bf16_1b.bmodel
#       ../models/BM1684X/pi05_ddn6_12_bf16_1b.bmodel
#       ../models/BM1684X/pi05_ddn12_18_bf16_1b.bmodel
#
# 为什么拆三段: 与 dkv 同理，动作专家 18 层单图在低比特下编译失败，按 6 层切分。
#
# 输入张量（17 个）:
#   suffix_in [1,10,32]  或 [1,10,1024]  —— 当前去噪步的动作隐变量（段 0–5 为 32 维，
#                                            段 6–11 / 12–17 为 1024 维，中间维度由模型定义决定）
#   time      [1]                        —— 当前去噪步的 flow-matching 时间
#   f4d       [1,1,10,546]               —— suffix 的注意力掩码（546 = 536 prefix + 10 suffix）
#   s_cos     [1,10,256]                 —— suffix RoPE cos
#   s_sin     [1,10,256]                 —— suffix RoPE sin
#   kv_*      [1,1,536,256] × 12         —— 由 dkv 段产出、设备常驻的 KV（本段负责的 6 层 × K/V）
# 输出: 段 0–5 / 6–12 输出 suffix_out（喂给下一段），段 12–17 输出 v_t（速度场）
#
# 去噪循环: 每个动作 chunk 需迭代 num_steps 次（本 sample 默认 dn2），每次迭代把三段
#   依次跑一遍，由宿主侧做 Euler 积分 x_{t+1} = x_t + dt * v_t。

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
KV="[1,1,$PL,256]"
KV12="$KV,$KV,$KV,$KV,$KV,$KV,$KV,$KV,$KV,$KV,$KV,$KV"

gen_mlir()
{
    # $1 = 模型名, $2 = suffix 输入维度, $3 = 输出张量名
    model_transform.py \
        --model_name $1 \
        --model_def ../models/onnx/${1}_fx.onnx \
        --input_shapes "[[1,10,$2],[1],[1,1,10,$((PL+10))],[1,10,256],[1,10,256],$KV12]" \
        --output_names $3 \
        --mlir $1.mlir
}

gen_bf16bmodel()
{
    model_deploy.py \
        --mlir $1.mlir \
        --quantize BF16 \
        --chip $target \
        --model ${1}_bf16_1b.bmodel
    mv ${1}_bf16_1b.bmodel $outdir/
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
gen_mlir pi05_ddn0_6       32   suffix_out
gen_mlir pi05_ddn6_12      1024 suffix_out
gen_mlir pi05_ddn12_18     1024 v_t
gen_bf16bmodel pi05_ddn0_6
gen_bf16bmodel pi05_ddn6_12
gen_bf16bmodel pi05_ddn12_18
popd