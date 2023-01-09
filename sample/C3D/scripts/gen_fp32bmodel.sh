#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))
if [ ! $1 ]; then
    target="BM1684X"
else
    target=$1
fi
outdir=../data/models/$target


function gen_fp32bmodel()
{
    python3 -m bmnetp --net_name=c3d \
                      --target=$target \
                      --opt=1 \
                      --cmp=true \
                      --shapes=[$1,3,16,112,112] \
                      --model=../data/models/c3d_ucf101.pt \
                      --outdir=$outdir \
                      --dyn=false
    mv $outdir/compilation.bmodel $outdir/c3d_fp32_$1b.bmodel

}

pushd $model_dir
#batch_size=1
gen_fp32bmodel 1
#batch_size=4
gen_fp32bmodel 4
popd