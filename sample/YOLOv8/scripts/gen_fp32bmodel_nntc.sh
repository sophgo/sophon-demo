#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))

if [ ! $1 ]; then
    echo "Please set the target chip. Option: BM1684 and BM1684X"
    exit
else
    target=$1
fi

outdir=../models/$target

function gen_fp32bmodel()
{
    python3 -m bmnetp  \
            --model=../models/torch/yolov8s.torchscript.pt \
            --target=$target \
            --shapes=[[$1,3,640,640]] \
            --net_name=yolov8s \
            --opt=1 \
            --dyn=False \
            --cmp=True \
            --enable_profile=True 
    mv compilation/compilation.bmodel $outdir/yolov8s_fp32_$1b.bmodel
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
gen_fp32bmodel 1

popd