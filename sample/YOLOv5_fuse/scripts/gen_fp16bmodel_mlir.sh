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

function gen_fp16bmodel()
{
    model_deploy.py \
        --mlir yolov5s_v6.1_3output_$1b.mlir \
        --quantize F16 \
        --chip  $target \
        --processor  $target \
        --fuse_preprocess \
        --customization_format BGR_PACKED \
        --model yolov5s_v6.1_3output_fp16_$1b.bmodel

    mv yolov5s_v6.1_3output_fp16_$1b.bmodel $outdir/
    if test $target = "bm1688";then
        model_deploy.py \
            --mlir yolov5s_v6.1_3output_$1b.mlir \
            --quantize F16 \
            --chip  $target \
            --processor  $target \
            --fuse_preprocess \
            --customization_format BGR_PACKED \
            --num_core 2 \
            --model yolov5s_v6.1_3output_fp16_$1b_2core.bmodel

        mv yolov5s_v6.1_3output_fp16_$1b_2core.bmodel $outdir/
    fi
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
gen_fp16bmodel 1

popd