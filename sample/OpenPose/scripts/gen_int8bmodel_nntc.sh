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
    python3 -m ufw.cali.cali_model  \
            --net_name=pose_coco  \
            --model ../models/caffe/pose/coco/pose_deploy_linevec.prototxt \
            --weights ../models/caffe/pose/coco/pose_iter_440000.caffemodel \
            --cali_image_path=../datasets/coco128  \
            --cali_iterations=128   \
            --cali_image_preprocess='resize_h=368,resize_w=368;mean_value=128.0:128.0:128.0,scale=0.003921569,bgr2rgb=True'   \
            --input_shapes="[1,3,368,368]"  \
            --target=$target   \
            --convert_bmodel_cmd_opt="-opt=2"   \
            --test_iterations 30 \

    mv ../models/caffe/pose/coco/pose_coco_batch1/compilation.bmodel $outdir/pose_coco_int8_1b.bmodel
}

function gen_int8bmodel()
{
    bmnetu --model=../models/caffe/pose/coco/pose_coco_bmnetc_deploy_int8_unique_top.prototxt \
        --weight=../models/caffe/pose/coco/pose_coco_bmnetc.int8umodel \
        -net_name=pose_coco \
        --shapes=[$1,3,368,368] \
        -target=$target \
        -opt=2
    mv compilation/compilation.bmodel $outdir/pose_coco_int8_$1b.bmodel
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