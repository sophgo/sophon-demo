#!/bin/bash
# INT8 BModel compilation script for ArcFace ResNet50
model_dir=$(dirname $(readlink -f "$0"))

if [ ! $1 ]; then
    target=bm1684x
    target_dir=BM1684X
else
    target=${1,,}
    target_dir=${target^^}
fi

outdir=../models/$target_dir

# tensor 输入不依赖烤入 do_preprocess(mean/scale)（当代工具链不 fuse 时不生效），
# 故用内嵌归一化节点的 onnx(w600k_r50_pre.onnx) + identity 预处理，三芯片统一喂原始像素。
# 校准方法：percentile9999（INT8 精度实测最优，min cos=0.9815）。
MODEL_DEF=../models/onnx/w600k_r50_pre.onnx
MEAN=0,0,0
SCALE=1,1,1

function gen_mlir()
{
    model_transform.py \
        --model_name arcface_resnet50 \
        --model_def $MODEL_DEF \
        --input_shapes [[$1,3,112,112]] \
        --mean $MEAN \
        --scale $SCALE \
        --pixel_format rgb \
        --mlir arcface_resnet50_$1b.mlir
}

function gen_cali_table()
{
    run_calibration.py arcface_resnet50_$1b.mlir \
        --dataset ../datasets/cali \
        --input_num 100 \
        --cali_method percentile9999 \
        -o arcface_cali_table_$1b
}

function gen_int8bmodel()
{
    model_deploy.py \
        --mlir arcface_resnet50_$1b.mlir \
        --quantize INT8 \
        --chip $target \
        --calibration_table arcface_cali_table_$1b \
        --model arcface_resnet50_int8_$1b.bmodel

    mv arcface_resnet50_int8_$1b.bmodel $outdir/
    if test $target = "bm1688";then
        model_deploy.py \
            --mlir arcface_resnet50_$1b.mlir \
            --quantize INT8 \
            --chip $target \
            --calibration_table arcface_cali_table_$1b \
            --model arcface_resnet50_int8_$1b_2core.bmodel \
            --num_core 2
        mv arcface_resnet50_int8_$1b_2core.bmodel $outdir/
    fi
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi

# batch_size=1
gen_mlir 1
gen_cali_table 1
gen_int8bmodel 1

# batch_size=4
gen_mlir 4
gen_cali_table 4
gen_int8bmodel 4

popd
