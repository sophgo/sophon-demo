#!/bin/bash
script_dir=$(dirname $(readlink -f "$0"))

if [ ! $1 ]; then
    echo "Please set the target chip. Option: BM1684 and BM1684X"
    exit
else
    target=$1
fi

outdir=../models/$target


function gen_fp32bmodel()
{
    python3 -m bmnetp --net_name=resnet50 \
                      --target=$target \
                      --opt=1 \
                      --cmp=true \
                      --shapes=[$1,3,224,224] \
                      --model=../models/torch/resnet50-11ad3fa6.torchscript.pt \
                      --outdir=$outdir \
                      --dyn=false
    if [ $? -ne 0 ]; then
        echo "gen_fp32bmodel batch_size $1 failed"
    else
        mv $outdir/compilation.bmodel $outdir/resnet50_fp32_$1b.bmodel
    fi

}

pushd $script_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
gen_fp32bmodel 1

popd
