#!/bin/bash
script_dir=$(dirname $(readlink -f "$0"))

if [ ! $1 ]; then
    target="BM1684X"
else
    target=$1
fi

outdir=../models/$target


function gen_fp32bmodel()
{
    python3 -m bmpaddle --net_name=fcenet \
                    --target=$target \
                    --opt=2 \
                    --cmp=true \
                    --shapes=[$1,3,736,992] \
                    --model=../models/paddle/inference/det_fce/ \
                    --outdir=$outdir \
                    --dyn=false \
                    --output_names="concat_0.tmp_0,concat_1.tmp_0,concat_2.tmp_0"
    if [ $? -ne 0 ]; then
        echo "gen_fp32bmodel batch_size $1 failed"
    else
        mv $outdir/compilation.bmodel $outdir/fcenet_fp32_b$1.bmodel
    fi

}

pushd $script_dir
#batch_size=1
gen_fp32bmodel 1
#batch_size=4
gen_fp32bmodel 4
popd
