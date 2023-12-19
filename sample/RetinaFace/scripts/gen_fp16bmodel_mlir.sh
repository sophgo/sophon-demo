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

outdir=../data/models/$target_dir

function gen_mlir()
{
    model_transform.py \
        --model_name retinaface_mobilenet0.25 \
        --model_def ../data/models/onnx/retinaface_mobilenet0.25.onnx \
        --input_shapes [[$1,3,640,640]] \
        --mlir retinaface_mobilenet0.25_$1b.mlir
}

function gen_fp16bmodel()
{
    model_deploy.py \
        --mlir retinaface_mobilenet0.25_$1b.mlir \
        --quantize F16 \
        --chip $target \
        --model retinaface_mobilenet0.25_fp16_$1b.bmodel

    mv retinaface_mobilenet0.25_fp16_$1b.bmodel $outdir/

    if test $target = "bm1688";then
        model_deploy.py \
            --mlir retinaface_mobilenet0.25_$1b.mlir \
            --quantize F16 \
            --chip $target \
            --model retinaface_mobilenet0.25_fp16_$1b_2core.bmodel \
            --num_core 2
        mv retinaface_mobilenet0.25_fp16_$1b_2core.bmodel $outdir/
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