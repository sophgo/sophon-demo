#!/bin/bash

if [ $# -lt 1 ];then
    echo "Erro: please input platform, eg: BM1684"
    popd
    exit -1
fi

platform=$1
echo "start fp32bmodel transform, platform: ${platform} ......"

root_dir=$(cd `dirname $BASH_SOURCE[0]`/../ && pwd)
build_dir=$root_dir/build
src_model_file=${root_dir}/data/models/torch/yolov5s_640_coco_v6.1_3output.torchscript.pt
src_model_name=`basename ${src_model_file}`
dst_model_prefix="yolov5s"
dst_model_postfix="coco_v6.1_3output"
fp32model_dir="${root_dir}/data/models/${platform}/fp32model"
int8model_dir="${root_dir}/data/models/${platform}/int8model"
lmdb_src_dir="${root_dir}/data/images"
image_src_dir="${root_dir}/data/images/coco200"
# lmdb_src_dir="${build_dir}/coco2017val/coco/images/"
#lmdb_dst_dir="${build_dir}/lmdb/"
img_size=${2:-640}
batch_size=${3:-1}
iteration=${4:-2}
img_width=640
img_height=640

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
