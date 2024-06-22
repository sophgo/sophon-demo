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
        --model_name yolov9s${opt} \
        --model_def ../models/onnx/yolov9s${opt}_$1b.onnx \
        --input_shapes [[$1,3,640,640]] \
        --mean 0.0,0.0,0.0 \
        --scale 0.0039216,0.0039216,0.0039216 \
        --keep_aspect_ratio \
        --pixel_format rgb  \
        --mlir yolov9s${opt}_$1b.mlir
}

function gen_fp32bmodel()
{
    model_deploy.py \
        --mlir yolov9s${opt}_$1b.mlir \
        --quantize F32 \
        --chip $target \
        --model yolov9s${opt}_fp32_$1b.bmodel

    mv yolov9s${opt}_fp32_$1b.bmodel $outdir/
    if test $target = "bm1688";then
        model_deploy.py \
            --mlir yolov9s${opt}_$1b.mlir \
            --quantize F32 \
            --chip $target \
            --model yolov9s${opt}_fp32_$1b_2core.bmodel \
            --num_core 2
            # --test_input ../datasets/test/3.jpg \
            # --test_reference yolov5_top.npz \
            # --debug
        mv yolov9s${opt}_fp32_$1b_2core.bmodel $outdir/
    fi
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
gen_mlir 1
gen_fp32bmodel 1
#opt="_opt"
#gen_mlir 1
#gen_fp32bmodel 1
popd
