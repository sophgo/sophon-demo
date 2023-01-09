#!/bin/bash
script_dir=$(dirname $(readlink -f "$0"))

if [ ! $1 ]; then
    target="BM1684X"
else
    target=$1
fi

outdir=../data/models/$target
cali_data_path=../data/c3d_lmdb/


function gen_int8bmodel()
{

    python3 -m ufw.cali.cali_model \
        --net_name 'c3d' \
        --model ../data/models/c3d_ucf101.pt \
        --test_iterations 1 \
        --cali_lmdb ${cali_data_path} \
        --input_shapes [1,3,16,112,112] \
        --try_cali_accuracy_opt="-fpfwd_inputs=40;-fpfwd_outputs=85;" \
        --convert_bmodel_cmd_opt "-outdir=$outdir --enable_profile True" \
        --cali_iterations 300 \
        --target ${target}
    if [ $? -ne 0 ]; then
        echo "gen_int8bmodel batch_size 1 failed"
    else
        mv $outdir/compilation.bmodel $outdir/c3d_int8_1b.bmodel
    fi
}

function gen_int8bmodel_b4()
{
    bmnetu -model ../data/models/c3d_bmnetp_deploy_int8_unique_top.prototxt \
       -weight ../data/models/c3d_bmnetp.int8umodel \
       -max_n 4 \
       -prec=INT8 \
       -dyn=0 \
       -cmp=1 \
       -target=$target \
       -outdir=$outdir
    if [ $? -ne 0 ]; then
        echo "gen_int8bmodel batch_size 4 failed"
    else
        mv $outdir/compilation.bmodel $outdir/c3d_int8_4b.bmodel
    fi
}

pushd $script_dir
# b1
gen_int8bmodel
# b4
gen_int8bmodel_b4
popd
