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
    python3 -m bmnetp \
        --net_name=centernet \
        --target=$target \
        --opt=2 \
        --cmp=true \
        --enable_profile=true \
        --shapes=[1,3,512,512] \
        --model=../models/torch/ctdet_coco_dlav0_1x.torchscript.pt \
        --dyn=false

    mv compilation/compilation.bmodel $outdir/centernet_fp32_1b.bmodel
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
gen_fp32bmodel 1
# batch_size=4
# gen_fp32bmodel 4
popd