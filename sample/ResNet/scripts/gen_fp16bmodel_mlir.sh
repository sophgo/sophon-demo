#!/bin/bash
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
        --model_name resnet50_$1b \
        --model_def ../models/torch/resnet50-11ad3fa6.torchscript.pt \
        --input_shapes [[$1,3,224,224]] \
        --keep_aspect_ratio \
        --pixel_format rgb  \
        --test_input ../datasets/cali_data/ILSVRC2012_val_00000555.jpg \
        --test_result resnet50_$1b_top_outputs.npz \
        --mlir resnet50_$1b.mlir
}

function gen_fp16bmodel()
{
    model_deploy.py \
        --mlir resnet50_$1b.mlir \
        --quantize F16 \
        --chip $target \
        --model resnet50_fp16_$1b.bmodel
        # --test_input resnet50_$1b_in_f32.npz \
        # --test_reference resnet50_$1b_top_outputs.npz

    mv resnet50_fp16_$1b.bmodel $outdir/
    if test $target = "bm1688";then
        model_deploy.py \
            --mlir resnet50_$1b.mlir \
            --quantize F16 \
            --chip $target \
            --model resnet50_fp16_$1b_2core.bmodel \
            --num_core 2
            # --test_input resnet50_$1b_in_f32.npz \
            # --test_reference resnet50_$1b_top_outputs.npz
        mv resnet50_fp16_$1b_2core.bmodel $outdir/
    fi
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
gen_mlir 1
gen_fp16bmodel 1

popd
