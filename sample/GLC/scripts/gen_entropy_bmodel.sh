#!/bin/bash
# 为 GLC 真熵编码子图编译 bmodel。
#
# 沿用现有 gen_bf16bmodel_mlir.sh 的模式，把子图换成真熵编码需要的五套：
#   glc_prior_init   z_hat                  -> q_enc q_dec scales means common
#   glc_stage_1/2/3  y_hat_so_far + common  -> scales means
#   glc_tail         y_hat + q_dec          -> x_hat
#
# 精度沿用现有结论：FP32 与 PyTorch 逐位一致，BF16 推荐，FP16 因激活溢出不可用。
# 默认 BF16，可用 `--fp32` 切到 FP32 做逐步对账。
#
# 用法：scripts/gen_entropy_bmodel.sh [bm1684x|bm1688] [--fp32]

set -euo pipefail

model_dir=$(dirname $(readlink -f "$0"))
proj_dir=$(dirname "$model_dir")

target=bm1684x
# quant 是 model_deploy.py 的 --quantize 取值（F32/BF16），suffix 是产物文件名的
# 精度后缀（fp32/bf16）。两者不同名：TPU-MLIR 的 F32 对应约定俗成的 "fp32" 文件名，
# 与 gen_fp32bmodel_mlir.sh 保持一致。
quant=BF16
suffix=bf16
for arg in "$@"; do
    case "$arg" in
        --fp32) quant=F32;  suffix=fp32 ;;
        --bf16) quant=BF16; suffix=bf16 ;;
        *) target=${arg,,} ;;
    esac
done
target_dir=${target^^}
outdir="$proj_dir/models/$target_dir"
onnx_dir="$proj_dir/models/onnx"

if [ ! -d "$onnx_dir" ]; then
    echo "缺少 ONNX 子图：$onnx_dir，先跑 export_subgraphs.py" >&2
    exit 1
fi

# 子图名 -> 输入 shape/type 清单，与 export_subgraphs.py 里造的示例输入保持一致
declare -A SHAPES=(
    [glc_prior_init]="[[1,256,8,12]]"
    [glc_stage_1]="[[1,256,32,48],[1,256,32,48]]"
    [glc_stage_2]="[[1,256,32,48],[1,256,32,48]]"
    [glc_stage_3]="[[1,256,32,48],[1,256,32,48]]"
    [glc_tail]="[[1,256,32,48],[1,256,1,1]]"
)
declare -A TYPES=(
    [glc_prior_init]="float32"
    [glc_stage_1]="float32,float32"
    [glc_stage_2]="float32,float32"
    [glc_stage_3]="float32,float32"
    [glc_tail]="float32,float32"
)
SUBS=(glc_prior_init glc_stage_1 glc_stage_2 glc_stage_3 glc_tail)

mkdir -p "$outdir"
workdir=$(mktemp -d)
trap 'rm -rf "$workdir"' EXIT

for name in "${SUBS[@]}"; do
    if [ ! -f "$onnx_dir/$name.onnx" ]; then
        echo "缺少 $onnx_dir/$name.onnx" >&2
        exit 1
    fi
    echo "=== $name ==="
    model_transform.py \
        --model_name "$name" \
        --model_def "$onnx_dir/$name.onnx" \
        --input_shapes "${SHAPES[$name]}" \
        --input_types "${TYPES[$name]}" \
        --mlir "$workdir/$name.mlir"

    model_deploy.py \
        --mlir "$workdir/$name.mlir" \
        --quantize "$quant" \
        --chip "$target" \
        --model "$workdir/${name}_${suffix}.bmodel"

    cp "$workdir/${name}_${suffix}.bmodel" "$outdir/"
    ls -la "$outdir/${name}_${suffix}.bmodel"
done

echo "完成：$outdir （精度 $suffix）"
