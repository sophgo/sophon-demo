#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))

if [ ! $1 ]; then
    target=bm1684x
else
    target=$1
fi

outdir=../models/BM1684X

function gen_mlir()
{
    model_transform.py \
        --model_name centernet \
        --model_def ../models/onnx/centernet_$1b.onnx \
        --input_shapes [[$1,3,512,512]] \
        --mean 104.01195,114.03422,119.91659 \
        --scale 0.014,0.014,0.014 \
        --keep_aspect_ratio \
        --pixel_format rgb  \
        --test_input ../datasets/ctdet_test.jpg \
        --test_result centernet_top_outputs.npz \
        --mlir centernet_$1b.mlir
}

function gen_fp32bmodel()
{
    model_deploy.py \
        --mlir centernet_$1b.mlir \
        --quantize F32 \
        --chip $target \
        --test_input centernet_in_f32.npz \
        --test_reference centernet_top_outputs.npz \
        --model centernet_fp32_$1b.bmodel \
        # --disable_layer_group 

    mv centernet_fp32_$1b.bmodel $outdir/
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
gen_mlir 1
gen_fp32bmodel 1

popd