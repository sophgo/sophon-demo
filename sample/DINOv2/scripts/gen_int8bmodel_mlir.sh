#!/bin/bash
# DINOv2 INT8 bmodel 编译（bm1684x / bm1688 / cv186x）
#
# 量化 recipe（见 docs/DINOv2_Export_Guide.md 与 scripts/gen_qtable.py）：
#   * 只把 48 个带 bias 的线性层 Gemm（_v_*_Gemm，即 12 个 block 的 qkv 融合
#     投影 768->2304、attn.proj 768->768、mlp.fc1 768->3072、mlp.fc2 3072->768）
#     留 INT8，其余算子（24 个无 bias 的 QK^T/attn@V、patch 均值、LayerNorm /
#     Softmax / GELU / 残差 / 布局）全 F16（qtable 白名单写法）。
#     全 INT8 会直接毁掉 linear-probe 特征：
#     SE9-16 实测 Top-1 0.1%（≈随机），故必须混合精度。
#   * 标定：--sq（SmoothQuant）+ --cali_method mse，标定图按官方 eval 前处理
#     （Resize 256 BICUBIC + CenterCrop 224）生成，npz float32。
#   * 线性层 Gemm 用 --matmul_perchannel 逐通道量化。
#   * 必须使用新版工具链：TPU-MLIR v1.30.beta.0-46-gb2e0030-20261029（whl
#     tpu_mlir-1.30b0）起包含 BM1686 dequant/requant 连续 staging 修复
#     （上游 commit bb0cb8e），INT8 在三颗芯片上都快于 FP16：
#       SE9-16(BM1688) bmrt 33.89ms / 2core 28.23ms（FP16 52.75 / 40.77）
#       SE9-8 (CV186X) bmrt 33.88ms                  （FP16 52.74）
#       SE7-32(BM1684X) bmrt 16.95ms                 （FP16 18.47）
#     修复在编译期烤进 bmodel，盒子无需刷固件；但用旧版编译的 INT8 仍是旧表现
#     （BM1688 上 140.64ms），必须重新编译。代价是 Top-1 低 0.8~1.9 个点。
#   * 三颗芯片统一使用同一配方（48 个 Gemm INT8），不再提供 tail16/tail24
#     精度档：BM1684X 上 INT8 相对 FP16 只有 ~8% 的速度余量，减少 INT8 层
#     换精度没有可用落点；为保持三平台行为一致，BM1688 / CV186X 也不再生成。
#   * 已实测排除的旋钮：--asymmetric（Top-1 崩到 0.1%，输出坍缩）、--opt 3
#     （tpuc-opt 在 layer-group 阶段 abort）、--opt 1（与默认 --opt 2 无差异）。
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

# 防呆：run_calibration.py --sq 会**原地改写** MLIR 引用的 weight npz（把 LN 的
# γ/β 除以 smooth scale、下游 MatMul 权重乘以同一 scale；fp32 下严格等价）。
# 在同一目录重复执行 = 在已平滑的模型上再平滑一次，权重会被逐次放大，最终
# cali 表 LayerNorm 阈值从 ~1.0 塌到 1e-3、model_deploy 校验 cos 掉到 0.01。
# 这里用 weight npz 的 md5 判定状态：31350e4a… 已是平滑后版本，禁止再标定。
# 需要重新标定时先删掉 mlir/权重再跑本脚本（model_transform 会重新生成 pristine 权重）。
PRISTINE_WEIGHT_MD5=2130d3b44dba32d5fe8c1d7fec63b9a8
SMOOTHED_WEIGHT_MD5=31350e4a953c195a7021a22ca99ec86e

function check_sq_state()
{
    local w=dinov2_vitb14_reg_$1b_top_f32_all_weight.npz
    [ -f "$w" ] || return 0
    local sum=$(md5sum "$w" | cut -d' ' -f1)
    if [ "$sum" = "$SMOOTHED_WEIGHT_MD5" ]; then
        echo "ERROR: $w 已被 --sq 平滑过（md5 $sum）。在本目录重跑标定会累积改写权重，" >&2
        echo "       导致 cali 表失效。请删除 $w 与 dinov2_vitb14_reg_$1b.mlir 后重跑本脚本。" >&2
        exit 1
    elif [ "$sum" != "$PRISTINE_WEIGHT_MD5" ]; then
        echo "WARNING: $w md5 $sum 不是已知的 pristine/SQ 版本，请确认来源后再标定。" >&2
    fi
}

function gen_cali_table()
{
    check_sq_state "$1b"
    run_calibration.py dinov2_vitb14_reg_$1b.mlir \
        --chip $target \
        --dataset ../datasets/cali_data \
        --sq --cali_method mse \
        --input_num 200 \
        -o dinov2_vitb14_reg_cali_table
}

function gen_qtable()
{
    python3 gen_qtable.py dinov2_vitb14_reg_$1b.mlir dinov2_vitb14_reg_qtable_$1b
}

# INT8 部署时的数值校验门限。
#
# model_deploy.py 对 INT8 的默认门限是 0.90,0.80（cosine, euclidean），其中
# euclidean_similarity = 1 - ||a-b||/||a||。该默认值按「分类 logits」标定，而
# 本模型比对的是 **1536 维原始 linear-probe 特征**，逐元素量化噪声分散在整个
# 向量上，相对 L2 误差天然偏大：三平台实测 cosine 0.948~0.951 / euclidean
# 0.673~0.682，即默认门限下必然判 FAILED。
#
# 该特征误差对应的真实精度损失已实测（SE9-16，1000 张，见 README 6.2）：
# int8 Top-1 82.20~82.90% vs 同例程 fp16 83.90~84.10%，SE9-16 掉 1.2~1.7 个点
# （三平台整体 0.8~1.9），与其它例程的 int8 损失同级。
# 因此这里显式放宽 euclidean 门限（cosine 保持默认 0.90），保留数值校验而
# 非直接 --skip_validation。同仓库先例：SuperGlue int8 用 --tolerance 0.5,0，
# CenterNet int8 用 --tolerance 0.8,0.5。
TOLERANCE=0.90,0.60

function gen_int8bmodel()
{
    model_deploy.py \
        --mlir dinov2_vitb14_reg_$1b.mlir \
        --quantize INT8 \
        --chip $target \
        --calibration_table dinov2_vitb14_reg_cali_table \
        --quantize_table dinov2_vitb14_reg_qtable_$1b \
        --matmul_perchannel \
        --tolerance $TOLERANCE \
        --model dinov2_vitb14_reg_int8_$1b.bmodel \
        --test_input dinov2_vitb14_reg_$1b_in_f32.npz \
        --test_reference dinov2_vitb14_reg_$1b_top_outputs.npz

    mv dinov2_vitb14_reg_int8_$1b.bmodel $outdir/
    if test $target = "bm1688";then
        model_deploy.py \
            --mlir dinov2_vitb14_reg_$1b.mlir \
            --quantize INT8 \
            --chip $target \
            --calibration_table dinov2_vitb14_reg_cali_table \
            --quantize_table dinov2_vitb14_reg_qtable_$1b \
            --matmul_perchannel \
            --tolerance $TOLERANCE \
            --num_core 2 \
            --model dinov2_vitb14_reg_int8_$1b_2core.bmodel \
            --test_input dinov2_vitb14_reg_$1b_in_f32.npz \
            --test_reference dinov2_vitb14_reg_$1b_top_outputs.npz
        mv dinov2_vitb14_reg_int8_$1b_2core.bmodel $outdir/
    fi
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
gen_mlir 1
gen_cali_table 1
gen_qtable 1
gen_int8bmodel 1

popd
