#!/bin/bash
script_dir=$(dirname $(readlink -f "$0"))

if [ ! $1 ]; then
    target="BM1684X"
else
    target=$1
fi

outdir=../data/models/$target


function gen_fp32bmodel()
{
    python3 -m bmnetp --net_name=resnet \
                      --target=$target \
                      --opt=1 \
                      --cmp=true \
                      --shapes=[$1,3,224,224] \
                      --model=../data/models/torch/resnet50-11ad3fa6_traced_b$1.pt \
                      --outdir=$outdir \
                      --dyn=false
    if [ $? -ne 0 ]; then
        echo "gen_fp32bmodel batch_size $1 failed"
    else
        mv $outdir/compilation.bmodel $outdir/resnet_fp32_b$1.bmodel
    fi

}

pushd $script_dir
#batch_size=1
gen_fp32bmodel 1
#batch_size=4
gen_fp32bmodel 4
popd
