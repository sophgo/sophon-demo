#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))

if [ ! $1 ]; then
    target=bm1684x
else
    target=$1
fi

outdir=../models/$target

function gen_mlir()
{
    model_transform.py \
        --model_name bert4torch_output \
        --model_def ../models/onnx/bert4torch_output.onnx \
        --input_shapes [[$1,256]] \
        --mlir bert4torch_output_$1b.mlir
}

function gen_fp16bmodel()
{
    model_deploy.py \
        --mlir bert4torch_output_$1b.mlir \
        --quantize F16 \
        --chip $target \
        --model bert4torch_output_fp16_$1b.bmodel

    mv bert4torch_output_fp16_$1b.bmodel $outdir/
    if test $target = "bm1688";then
        model_deploy.py \
            --mlir bert4torch_output_$1b.mlir \
            --quantize F16 \
            --chip $target \
            --model bert4torch_output_fp16_$1b_2core.bmodel \
            --num_core 2
          
        mv bert4torch_output_fp16_$1b_2core.bmodel $outdir/
    fi
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
# gen_mlir 1
# gen_fp16bmodel 1
gen_mlir 8
gen_fp16bmodel 8
popd