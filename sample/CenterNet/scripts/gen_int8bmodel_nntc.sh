#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))

if [ ! $1 ]; then
    echo "Please set the target chip. Option: BM1684 and BM1684X"
    exit
else
    target=$1
fi

outdir=../models/$target

function auto_cali()
{
   
    python3 -m ufw.cali.cali_model \
             --net_name 'centernet' \
             --model ../models/torch/ctdet_coco_dlav0_1x.torchscript.pt \
             --cali_image_path ../datasets/coco128 \
             --cali_image_preprocess='resize_h=512,resize_w=512;mean_value=104.01195:114.03422:119.91659, scale=0.014' \
             --input_shapes [1,3,512,512] \
             --fp32_layer_list '30,33,36' \
             --target $target
    mv ../models/torch/centernet_batch1/compilation.bmodel $outdir/centernet_int8_1b.bmodel

}

function gen_int8bmodel()
{
    bmnetu --model=../models/torch/centernet_bmnetp_deploy_int8_unique_top.prototxt  \
           --weight=../models/torch/centernet_bmnetp.int8umodel \
           -net_name=centernet \
           --shapes=[$1,3,512,512] \
           -target=$target \
           -opt=1
    mv compilation/compilation.bmodel $outdir/centernet_int8_$1b.bmodel
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
auto_cali
# batch_size=4
gen_int8bmodel 4

popd