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
            --model=../models/torch/extractor.pt \
            --target=$target \
            --shapes=[[$1,3,128,64]] \
            --net_name=extractor \
            --opt=1 \
            --dyn=False \
            --cmp=True \
            --enable_profile=True 
    mv compilation/compilation.bmodel $outdir/extractor_fp32_$1b.bmodel
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
gen_fp32bmodel 1
batch_size=4
gen_fp32bmodel 4
popd