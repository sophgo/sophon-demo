#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))

if [ ! $1 ]; then
    echo "Please set the target chip. Option: BM1684 and BM1684X"
    exit
else
    target=$1
fi

outdir=../models/$target

function create_lmdb()
{
    rm ../datasets/coco128_lmdb/*
    # convert_imageset.py包含所有预处理流程
    python3 ../tools/convert_imageset.py \
            --imageset_rootfolder=../datasets/coco128 \
            --imageset_lmdbfolder=../datasets/coco128_lmdb/ \
            --resize_height=368 \
            --resize_width=368 \
            --shuffle=True \
            --bgr2rgb=True \
            --gray=False
}
function gen_fp32umodel()
{
    python3 -m ufw.tools.cf_to_umodel \
            -m '../models/caffe/pose/coco/pose_deploy_linevec.prototxt' \
            -w '../models/caffe/pose/coco/pose_iter_440000.caffemodel' \
            -s '(1,3,368,368)' \
            -d 'compilation' \
            -D '../datasets/coco128_lmdb' \
            --cmp
}
function gen_int8umodel()
{
    calibration_use_pb quantize \
            -model="compilation/pose_iter_440000_bmnetc_test_fp32.prototxt"   \
            -weights="compilation/pose_iter_440000_bmnetc.fp32umodel"  \
            -iterations=128 
}
function gen_int8bmodel()
{
    bmnetu --model=compilation/pose_iter_440000_bmnetc_deploy_int8_unique_top.prototxt \
           --weight=compilation/pose_iter_440000_bmnetc.int8umodel \
           --shapes=[$1,3,368,368] \
           --target=$target \
           --net_name='pose_coco' \
           --cmp=1 \
           --opt=2 

    mv compilation/compilation.bmodel $outdir/pose_coco_int8_$1b.bmodel
}

pushd $model_dir

create_lmdb
gen_fp32umodel
gen_int8umodel
# 1b
gen_int8bmodel 1
# 4b
gen_int8bmodel 4
popd