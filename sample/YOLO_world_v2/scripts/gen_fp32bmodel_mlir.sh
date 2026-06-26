#!/bin/bash
# ==============================================================================
# YOLO-World v2 FP32 BModel 编译 (TPU-MLIR)
# 在 flh_mlir 容器内 source envsetup.sh 后运行: bash gen_fp32bmodel_mlir.sh [bm1684x]
# 产物: yoloworld_v2_fp32_1b.bmodel, clip_text_vitb32_${target}_f16_1b.bmodel
# ==============================================================================
model_dir=$(dirname $(readlink -f "$0"))

if [ ! $1 ]; then
    target=bm1684x
    target_dir=BM1684X
else
    target=${1,,}
    target_dir=${target^^}
    if test $target = "bm1684"
    then
        echo "bm1684 do not support fp32"
        exit
    fi
fi

outdir=../models/$target_dir

function gen_mlir()
{
    model_transform.py \
        --model_name yoloworld_v2 \
        --model_def ../models/onnx/yoloworld_v2.onnx \
        --input_shapes [[$1,3,640,640],[1,80,512]] \
        --output_names output \
        --keep_aspect_ratio \
        --pixel_format rgb \
        --mlir yoloworld_v2_$1b.mlir
}

function gen_fp32bmodel()
{
    model_deploy.py \
        --mlir yoloworld_v2_$1b.mlir \
        --quantize F32 \
        --chip $target \
        --model yoloworld_v2_fp32_$1b.bmodel

    mv yoloworld_v2_fp32_$1b.bmodel $outdir/
    if test $target = "bm1688";then
        model_deploy.py \
            --mlir yoloworld_v2_$1b.mlir \
            --quantize F32 \
            --chip $target \
            --model yoloworld_v2_fp32_$1b_2core.bmodel \
            --num_core 2
        mv yoloworld_v2_fp32_$1b_2core.bmodel $outdir/
    fi
}

function gen_text_encoder_mlir()
{
    model_transform.py \
      --model_name clip_text_vitb32 \
      --model_def ../models/onnx/clip_text_vitb32.onnx \
      --input_shapes [[$1,77]] \
      --pixel_format rgb \
      --mlir clip_text_vitb32_$1b.mlir
}

function gen_text_encoder_fp16bmodel()
{
    model_deploy.py \
        --mlir clip_text_vitb32_$1b.mlir \
        --quantize F16 \
        --chip $target \
        --model clip_text_vitb32_${target}_f16_$1b.bmodel

    mv clip_text_vitb32_${target}_f16_$1b.bmodel $outdir/
    if test $target = "bm1688";then
        model_deploy.py \
            --mlir clip_text_vitb32_$1b.mlir \
            --quantize F16 \
            --chip $target \
            --model clip_text_vitb32_${target}_f16_$1b_2core.bmodel \
            --num_core 2
        mv clip_text_vitb32_${target}_f16_$1b_2core.bmodel $outdir/
    fi
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1 主检测模型 FP32
gen_mlir 1
gen_fp32bmodel 1

# batch_size=1 文本编码模型 FP16 (文本编码精度不敏感, 统一 FP16)
gen_text_encoder_mlir 1
gen_text_encoder_fp16bmodel 1
popd
