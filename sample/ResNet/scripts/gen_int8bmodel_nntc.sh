#!/bin/bash
script_dir=$(dirname $(readlink -f "$0"))

if [ ! $1 ]; then
    echo "Please set the target chip. Option: BM1684 and BM1684X"
    exit
else
    target=$1
fi

outdir=../models/$target
cali_data_path=../datasets/cali_data


function auto_cali()
{
    python3 -m ufw.cali.cali_model \
        --net_name=resnet50 \
        --model=../models/torch/resnet50-11ad3fa6.torchscript.pt \
        --cali_image_path=$cali_data_path \
        --cali_image_preprocess='resize_h=224,resize_w=224;mean_value=103.53:116.28:123.675,scale=0.01742919:0.017507:0.01712475,bgr2rgb=True' \
        --input_shapes="[1,3,224,224]" \
        --target=$target \
        --convert_bmodel_cmd_opt="-outdir=$outdir --target=$target -input_as_fp32=input.1 -opt=2 -v=4"

    if [ $? -ne 0 ]; then
        echo "gen_int8bmodel batch_size 1 failed"
    else
        mv $outdir/compilation.bmodel $outdir/resnet50_int8_1b.bmodel
    fi
}

function gen_int8bmodel()
{
    bmnetu --model=../models/torch/resnet50_bmnetp_deploy_int8_unique_top.prototxt  \
           --weight=../models/torch/resnet50_bmnetp.int8umodel \
           -net_name=resnet50 \
           --shapes=[$1,3,224,224] \
           -outdir=$outdir \
           -target=$target \
           -input_as_fp32=input.1 \
           -opt=2 \
           -v=4
    mv $outdir/compilation.bmodel $outdir/resnet50_int8_$1b.bmodel
}

pushd $script_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi

# batch_size=1
auto_cali
# batch_size=4
gen_int8bmodel 4
popd