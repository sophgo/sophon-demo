#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))
target=$1
outdir=../data/models/$target


function gen_fp32bmodel()
{
    bmnetc --net_name=ssd \
                      --target=$target \
                      --opt=1 \
                      --cmp=true \
                     --shapes=[$1,3,300,300] \
                      --model=../data/models/caffe/ssd300_deploy.prototxt \
                      --weight=../data/models/caffe/ssd300.caffemodel \
                      --outdir=$outdir \
                      --dyn=false
    mv $outdir/compilation.bmodel $outdir/ssd300_fp32_$1b.bmodel

}

pushd $model_dir
#batch_size=1
gen_fp32bmodel 1
#batch_size=4
gen_fp32bmodel 4
popd