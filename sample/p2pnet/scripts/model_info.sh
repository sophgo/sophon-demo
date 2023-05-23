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
src_model_file=${root_dir}/models/torch/p2pnet_trace.pt
src_model_name=`basename ${src_model_file}`
dst_model_prefix="p2pnet"
fp32model_dir="${root_dir}/models/${platform}/fp32model"
int8model_dir="${root_dir}/models/${platform}/int8model"
lmdb_src_dir="${root_dir}/data/lmdb"
image_src_dir="${root_dir}/data/ShanghaiTech/ShanghaiTech-Dataset/ShanghaiTech/part_A/test_data/images"
img_size=${2:-512}
batch_size=${3:-1}
iteration=${4:-2}
img_width=512
img_height=512

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
