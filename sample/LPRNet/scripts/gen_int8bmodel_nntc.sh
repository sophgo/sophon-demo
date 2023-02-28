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
    rm ../datasets/test_md5_lmdb/*
    # convert_imageset.py包含所有预处理流程
    python3 ../tools/convert_imageset.py \
        --imageset_rootfolder=../datasets/test_md5 \
        --imageset_lmdbfolder=../datasets/test_md5_lmdb/ \
        --resize_height=24 \
        --resize_width=94 \
        --shuffle=True \
        --bgr2rgb=True \
        --gray=False
}
function gen_fp32umodel()
{
    python3 -m ufw.tools.pt_to_umodel \
        -m ../models/torch/LPRNet_model_trace.pt \
        -s "(1, 3, 24, 94)" \
        -d compilation \
        -D ../datasets/test_md5_lmdb/ \
        --cmp
}
function gen_int8umodel()
{
    calibration_use_pb quantize \
        -model=compilation/LPRNet_model_trace_bmnetp_test_fp32.prototxt \
        -weights=compilation/LPRNet_model_trace_bmnetp.fp32umodel \
        -iterations=450 \
        -fpfwd_blocks="x.1,237"
}
function gen_int8bmodel()
{
    bmnetu -model=compilation/LPRNet_model_trace_bmnetp_deploy_int8_unique_top.prototxt \
           -weight=compilation/LPRNet_model_trace_bmnetp.int8umodel \
           --target=$target \
           -outdir=$outdir \
           -shapes=[$1,3,24,94] \
           -cmp false

    mv $outdir/compilation.bmodel $outdir/lprnet_int8_$1b.bmodel
}


pushd $model_dir

if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi

#在制作lmdb过程中使用bm_opencv
# export PYTHONPATH=$PYTHONPATH:$REL_TOP/lib/opencv/pcie/opencv-python/
# create_lmdb
gen_fp32umodel
gen_int8umodel
# 1b
gen_int8bmodel 1
# 4b
gen_int8bmodel 4
# tpu_model --combine $outdir/lprnet_int8_1b.bmodel $outdir/lprnet_int8_4b.bmodel -o $outdir/lprnet_int8_1b4b.bmodel 
popd