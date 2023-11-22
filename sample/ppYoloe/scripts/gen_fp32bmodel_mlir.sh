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
        --model_name ppyoloe \
        --model_def ../models/onnx/ppyoloe.onnx \
        --input_shapes [[$1,3,640,640],[$1,2]] \
        --mean 123.675,116.28,103.53 \
        --scale 0.0171,0.0175,0.0174 \
        --pixel_format rgb  \
        --output_names p2o.Div.1,p2o.Concat.29 \
        --mlir ppyoloe_$1b.mlir
}

function gen_fp32bmodel()
{
    model_deploy.py \
        --mlir ppyoloe_$1b.mlir \
        --quantize F32 \
        --chip ${target} \
        --model ppyoloe_fp32_$1b.bmodel

    mv ppyoloe_fp32_$1b.bmodel $outdir/
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi

# batch_size=1
gen_mlir 1
gen_fp32bmodel 1

# batch_size=4
# gen_mlir 4
# gen_fp32bmodel 4


popd