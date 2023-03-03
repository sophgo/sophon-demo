#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))

if [ ! $1 ]; then
    target=bm1684x
else
    target=$1
fi

outdir=../models/BM1684X

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

function gen_cali_table_coco()
{
    run_calibration.py pose_coco_$1b.mlir \
        --dataset ../datasets/coco128/ \
        --input_num 128 \
        -o pose_coco_cali_table
}

function gen_int8bmodel_coco()
{
    model_deploy.py \
        --mlir pose_coco_$1b.mlir \
        --quantize INT8 \
        --chip bm1684x \
        --asymmetric \
        --calibration_table pose_coco_cali_table \
        --model pose_coco_int8_$1b.bmodel

    mv pose_coco_int8_$1b.bmodel $outdir/
}


pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
gen_mlir_coco 1
gen_cali_table_coco 1
gen_int8bmodel_coco 1

# batch_size=4
gen_mlir_coco 4
gen_int8bmodel_coco 4

popd