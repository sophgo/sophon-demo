#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))

if [ ! $1 ]; then
    target=bm1684x
    target_dir=BM1684X
else
    target=${1,,}
    target_dir=${target^^}
fi

outdir=../models/$target_dir

function gen_mlir_body_25()
{   
    model_transform.py \
        --model_name pose_body_25 \
        --model_def ../models/caffe/pose/body_25/pose_deploy.prototxt \
        --model_data ../models/caffe/pose/body_25/pose_iter_584000.caffemodel \
        --input_shapes [[$1,3,368,368]] \
        --mean 128.0,128.0,128.0 \
        --scale 0.0039216,0.0039216,0.0039216 \
        --pixel_format rgb  \
        --mlir pose_body_25_$1b.mlir
}

function gen_fp32bmodel_body_25()
{
    model_deploy.py \
        --mlir pose_body_25_$1b.mlir \
        --quantize F32 \
        --chip $target \
        --model pose_body_25_fp32_$1b.bmodel

    mv pose_body_25_fp32_$1b.bmodel $outdir/
}

function gen_mlir_coco()
{   
    model_transform.py \
        --model_name pose_coco \
        --model_def ../models/caffe/pose/coco/pose_deploy_linevec.prototxt \
        --model_data ../models/caffe/pose/coco/pose_iter_440000.caffemodel \
        --input_shapes [[$1,3,368,368]] \
        --mean 128.0,128.0,128.0 \
        --scale 0.0039216,0.0039216,0.0039216 \
        --pixel_format rgb  \
        --mlir pose_coco_$1b.mlir
}

function gen_fp32bmodel_coco()
{
    model_deploy.py \
        --mlir pose_coco_$1b.mlir \
        --quantize F32 \
        --chip $target \
        --model pose_coco_fp32_$1b.bmodel

    mv pose_coco_fp32_$1b.bmodel $outdir/
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
gen_mlir_coco 1
gen_fp32bmodel_coco 1

# batch_size=1
gen_mlir_body_25 1
gen_fp32bmodel_body_25 1
popd