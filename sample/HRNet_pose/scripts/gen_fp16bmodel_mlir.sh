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
        --model_name hrnetf16 \
        --model_def ../models/onnx/pose_hrnet_w32_256x192.onnx \
        --input_shapes [[1,3,256,192]] \
        --mean 123.675,123.675,103.53 \
        --scale 0.01712475,0.017507,0.01742919 \
        --keep_aspect_ratio \
        --pixel_format rgb \
        --test_input ../mlir_utils/person.jpg \
        --test_result hrnet_top_outputs.npz \
        --mlir hrnetf16.mlir
}

function gen_fp16bmodel()
{
    model_deploy.py \
        --mlir hrnetf16.mlir \
        --quantize F16 \
        --chip $target \
        --test_input hrnetf16_in_f32.npz \
        --test_reference hrnet_top_outputs.npz \
        --model hrnet_w32_256x192_f16.bmodel

    mv hrnet_w32_256x192_f16.bmodel $outdir/
    if test $target = "bm1688";then
        model_deploy.py \
            --mlir hrnetf16.mlir \
            --quantize F16 \
            --chip $target \
            --test_input hrnetf16_in_f32.npz \
            --test_reference hrnet_top_outputs.npz \
            --num_core 2 \
            --model hrnet_w32_256x192_f16_2core.bmodel
            

        mv hrnet_w32_256x192_f16_2core.bmodel $outdir/
    fi
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
gen_mlir
gen_fp16bmodel

popd