#!/bin/bash
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
        --model_name mobilenetv4_conv_medium_$1b \
        --model_def ../models/torch/mobilenetv4_conv_medium.torchscript.pt \
        --input_shapes [[$1,3,224,224]] \
        --keep_aspect_ratio \
        --pixel_format rgb  \
        --test_input ../datasets/cali_data/ILSVRC2012_val_00000555.jpg \
        --test_result mobilenetv4_conv_medium_$1b_top_outputs.npz \
        --mlir mobilenetv4_conv_medium_$1b.mlir
}

function gen_fp32bmodel()
{
    model_deploy.py \
        --mlir mobilenetv4_conv_medium_$1b.mlir \
        --quantize F32 \
        --chip $target \
        --model mobilenetv4_conv_medium_fp32_$1b.bmodel \
        --test_input mobilenetv4_conv_medium_$1b_in_f32.npz \
        --test_reference mobilenetv4_conv_medium_$1b_top_outputs.npz

    mv mobilenetv4_conv_medium_fp32_$1b.bmodel $outdir/
    if test $target = "bm1688";then
        model_deploy.py \
            --mlir mobilenetv4_conv_medium_$1b.mlir \
            --quantize F32 \
            --chip $target \
            --model mobilenetv4_conv_medium_fp32_$1b_2core.bmodel \
            --num_core 2 \
            --test_input mobilenetv4_conv_medium_$1b_in_f32.npz \
            --test_reference mobilenetv4_conv_medium_$1b_top_outputs.npz
        mv mobilenetv4_conv_medium_fp32_$1b_2core.bmodel $outdir/
    fi
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
gen_mlir 1
gen_fp32bmodel 1

popd
