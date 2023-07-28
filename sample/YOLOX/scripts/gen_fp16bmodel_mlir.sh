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
        --model_name yolox_s \
        --model_def ../models/onnx/yolox_s.onnx \
        --input_shapes [[$1,3,640,640]] \
        --mean 0.0,0.0,0.0 \
        --scale 0.0039216,0.0039216,0.0039216 \
        --keep_aspect_ratio \
        --pixel_format rgb  \
        --mlir yolox_s_$1b.mlir
}

function gen_fp16bmodel()
{
    model_deploy.py \
        --mlir yolox_s_$1b.mlir \
        --quantize F16 \
        --chip $target \
        --model yolox_s_fp16_$1b.bmodel

    mv yolox_s_fp16_$1b.bmodel $outdir/
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
gen_mlir 1
gen_fp16bmodel 1

# batch_size=4
# gen_mlir 4
# gen_fp16bmodel 4

popd