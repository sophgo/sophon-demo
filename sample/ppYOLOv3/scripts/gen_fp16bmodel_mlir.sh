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
        --model_name ppyolov3 \
        --model_def ../models/onnx/ppyolov3_$1b.onnx \
        --input_shapes [[$1,3,608,608]] \
        --mean 123.675,116.28,103.53 \
        --output_names Transpose_0,Transpose_2,Transpose_4 \
        --scale 0.01712475,0.017507,0.0174292 \
        --mlir ppyolov3_$1b.mlir
}

function gen_fp16bmodel()
{
    model_deploy.py \
        --mlir ppyolov3_$1b.mlir \
        --quantize F16 \
        --chip $target \
        --model ppyolov3_fp16_$1b.bmodel

    mv ppyolov3_fp16_$1b.bmodel $outdir/

    if test $target = "bm1688";then
        model_deploy.py \
            --mlir ppyolov3_$1b.mlir \
            --quantize F16 \
            --chip $target \
            --num_core 2 \
            --model ppyolov3_fp16_$1b_2core.bmodel
    
        mv ppyolov3_fp16_$1b_2core.bmodel $outdir/
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