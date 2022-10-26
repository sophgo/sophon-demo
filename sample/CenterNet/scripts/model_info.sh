#!/bin/bash
root_dir=$(cd `dirname $BASH_SOURCE[0]`/../data/ && pwd)
build_dir=$root_dir/build
pth_model_name=$root_dir/models/torch/ctdet_coco_dlav0_1x.pth
echo $pth_model_name
src_model_file="ctdet_coco_dlav0_1x.torchscript.pt"
src_model_path=$build_dir/$src_model_file
lmdb_src_dir=$root_dir/images
lmdb_dst_dir=$root_dir/images
img_size=512
int8model_dir="$build_dir/int8model"
fp32model_dir="$build_dir/fp32model_1684"
dst_model_prefix="ctdet_coco_dlav0_1output"
iteration=200

if [ ! -n $REL_TOP ]; then
    REL_TOP=/workspace
fi


function check_file()
{
    if [ ! -f $1 ]; then
        echo "$1 not exist."
        exit 1
    fi
}

function check_dir()
{
    if [ ! -d $1 ]; then
        echo "$1 not exist."
        exit 1
    fi
}
