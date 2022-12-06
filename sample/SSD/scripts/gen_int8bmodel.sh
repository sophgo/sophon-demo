#!/bin/bash
script_dir=$(dirname $(readlink -f "$0"))

if [ ! $1 ]; then
    target="BM1684X"
else
    target=$1
fi

outdir=../data/models/$target
cali_data_path=../data/images/lmdb/


function gen_int8bmodel()
{

    python3 -m ufw.cali.cali_model \
        --net_name 'ssd' \
        --model ../data/models/caffe/ssd300_deploy.prototxt \
        --weight ../data/models/caffe/ssd300.caffemodel \
        --cali_lmdb ${cali_data_path} \
        --input_shapes [1,3,300,300] \
        --cali_iterations=200 \
        --convert_bmodel_cmd_opt "-outdir=$outdir --target=$target --enable_profile=true"
    if [ $? -ne 0 ]; then
        echo "gen_int8bmodel batch_size 1 failed"
    else
        mv $outdir/compilation.bmodel $outdir/ssd300_int8_1b.bmodel
    fi
}

function gen_int8bmodel_b4()
{
    bmnetu -model ../data/models/caffe/ssd_bmnetc_deploy_int8_unique_top.prototxt \
       -weight ../data/models/caffe/ssd_bmnetc.int8umodel \
       -max_n 4 \
       -prec=INT8 \
       -dyn=0 \
       -cmp=1 \
       -target=$target \
       -outdir=$outdir
    if [ $? -ne 0 ]; then
        echo "gen_int8bmodel batch_size 4 failed"
    else
        mv $outdir/compilation.bmodel $outdir/ssd300_int8_4b.bmodel
    fi
}
pushd $script_dir
# b1
gen_int8bmodel
# b4
gen_int8bmodel_b4
popd
