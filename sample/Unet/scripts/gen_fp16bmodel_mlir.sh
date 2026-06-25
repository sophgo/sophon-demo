#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))
#默认为bm1684x
if [ ! $1 ]; 
then
    target=bm1684x
    target_dir=BM1684X
else
    target=${1,,}
    target_dir=${target^^}
fi

outdir=../models/$target_dir

function gen_mlir(){
    model_transform.py \
        --model_name unet \
        --model_def ../models/onnx/unet.onnx\
        --input_shapes [[$1,3,640,959]]  \
        --keep_aspect_ratio \
        --mlir unet_$1b.mlir 
}

function gen_fp32bmodel(){
    model_deploy.py \
        --mlir unet_$1b.mlir \
        --quantize F16 \
        --chip $target \
        --model unet_fp16_$1b.bmodel
    mv unet_fp16_$1b.bmodel $outdir/
}

# 从当前目录进入modol_dir
pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi

# batch_size=1
gen_mlir 1
gen_fp32bmodel 1

# 回到当前目录
popd
