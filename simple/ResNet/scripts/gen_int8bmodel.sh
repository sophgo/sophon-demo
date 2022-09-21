#!/bin/bash
script_dir=$(dirname $(readlink -f "$0"))

if [ ! $1 ]; then
    target="BM1684X"
else
    target=$1
fi

outdir=../data/models/$target
cali_data_path=../data/images/cali_data


function gen_int8bmodel()
{

    python3 -m ufw.cali.cali_model \
        --net_name 'resnet' \
        --model ../data/models/torch/resnet50-11ad3fa6_traced_b$1.pt \
        --cali_image_path $cali_data_path \
        --cali_image_preprocess='resize_h=224,resize_w=224;mean_value=103.53:116.28:123.675,scale=0.01742919:0.017507:0.01712475,bgr2rgb=True' \
        --input_shapes [$1,3,224,224] \
        --convert_bmodel_cmd_opt "-outdir $outdir --target $target -input_as_fp32=input.1 -opt=2 -v=4"

    if [ $? -ne 0 ]; then
        echo "gen_int8bmodel batch_size $1 failed"
    else
        mv $outdir/compilation.bmodel $outdir/resnet_int8_b$1.bmodel
    fi
}


pushd $script_dir
# b1
gen_int8bmodel 1
# b4
gen_int8bmodel 4
popd
