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
    python3 -m bmnetp --net_name=lprnet \
                      --target=$target \
                      --opt=1 \
                      --cmp=true \
                      --shapes=[$1,3,24,94] \
                      --model=../models/torch/LPRNet_model_trace.pt \
                      --outdir=$outdir \
                      --dyn=false
    mv $outdir/compilation.bmodel $outdir/lprnet_fp32_$1b.bmodel

}

pushd $model_dir

if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi

#batch_size=1
gen_fp32bmodel 1
#batch_size=4
# gen_fp32bmodel 4
# tpu_model --combine $outdir/lprnet_fp32_1b.bmodel $outdir/lprnet_fp32_4b.bmodel -o $outdir/lprnet_fp32_1b4b.bmodel 
popd