#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))

if [ ! $1 ]; then
    echo "Please set the target chip. Option: BM1684 and BM1684X"
    exit
else
    target=$1
fi

outdir=../models/$target

function gen_fp32bmodel_body_25()
{
    bmnetc --net_name=pose_body_25 \
           --target=$target \
           --cmp=true \
           --shapes=[$1,3,368,368] \
           --model=../models/caffe/pose/body_25/pose_deploy.prototxt \
           --weight=../models/caffe/pose/body_25/pose_iter_584000.caffemodel \
           --outdir=$outdir \
           --dyn=false
    mv $outdir/compilation.bmodel $outdir/pose_body_25_fp32_$1b.bmodel

}

function gen_fp32bmodel_coco()
{
    bmnetc --net_name=pose_coco \
           --target=$target \
           --cmp=true \
           --shapes=[$1,3,368,368] \
           --model=../models/caffe/pose/coco/pose_deploy_linevec.prototxt \
           --weight=../models/caffe/pose/coco/pose_iter_440000.caffemodel \
           --outdir=$outdir \
           --dyn=false
    mv $outdir/compilation.bmodel $outdir/pose_coco_fp32_$1b.bmodel

}

pushd $model_dir
# batch_size=1
gen_fp32bmodel_body_25 1
# batch_size=4
# gen_fp32bmodel_body_25 4
# batch_size=1
gen_fp32bmodel_coco 1
# batch_size=4
# gen_fp32bmodel_coco 4
popd