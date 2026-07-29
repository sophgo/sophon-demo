#!/bin/bash
# ==============================================================================
# YOLO-World v2 INT8 BModel 编译 (TPU-MLIR)
# 在 TPU-MLIR 容器内运行(参考 ../../docs/Environment_Install_Guide.md): bash gen_int8bmodel_mlir.sh [bm1684x]
# 产物: yoloworld_v2_int8_1b.bmodel, clip_text_vitb32_${target}_f16_1b.bmodel
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
        echo "bm1684 do not support fp16"
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
        --mean 0.0,0.0,0.0 \
        --scale 0.0039216,0.0039216,0.0039216 \
        --output_names output \
        --keep_aspect_ratio \
        --pixel_format rgb \
        --mlir yoloworld_v2_$1b.mlir
}

function gen_cali_table()
{
    run_calibration.py yoloworld_v2_$1b.mlir \
        --dataset ../datasets/cali_npz/ \
        --input_num 128 \
        --cali_method kl \
        -o yoloworld_v2_cali_table
}

function gen_int8bmodel()
{
    model_deploy.py \
        --mlir yoloworld_v2_$1b.mlir \
        --quantize INT8 \
        --chip $target \
        --calibration_table yoloworld_v2_cali_table \
        --quantize_table yoloworld_v2_qtable \
        --model yoloworld_v2_int8_$1b.bmodel

    mv yoloworld_v2_int8_$1b.bmodel $outdir/
    if test $target = "bm1688";then
        model_deploy.py \
            --mlir yoloworld_v2_$1b.mlir \
            --quantize INT8 \
            --chip $target \
            --calibration_table yoloworld_v2_cali_table \
            --quantize_table yoloworld_v2_qtable \
            --model yoloworld_v2_int8_$1b_2core.bmodel \
            --num_core 2
        mv yoloworld_v2_int8_$1b_2core.bmodel $outdir/
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
# batch_size=1 主检测模型 INT8
gen_mlir 1
gen_cali_table 1
gen_int8bmodel 1

# batch_size=1 文本编码模型 FP16
gen_text_encoder_mlir 1
gen_text_encoder_fp16bmodel 1
popd
