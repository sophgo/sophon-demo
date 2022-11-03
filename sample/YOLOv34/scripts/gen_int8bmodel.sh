#!/bin/bash

script_dir=$(dirname $(readlink -f "$0"))
if [ ! $1 ]; then
    target="BM1684"
else
    target=$1
fi
root_dir=$script_dir/..
data_dir=$root_dir/data
outdir=$data_dir/models/$target
image_src_dir=$data_dir/images/coco200
function gen_int8bmodel()
{
    python3 -m ufw.cali.cali_model  \
	    --net_name=yolov4_416_coco  \
            --model=../data/models/darknet/yolov4.cfg  \
	        --weight=../data/models/darknet/yolov4.weights \
    	    --cali_image_path=${image_src_dir}  \
    	    --cali_image_preprocess='resize_h=416,resize_w=416;scale=0.003921569,bgr2rgb=True'   \
            --input_shapes="[$1,3,416,416]"  \
    	    --outdir=$outdir   \
            --target=$target      
    if [ $? -ne 0 ]; then
        echo "gen_int8bmodel batch_size $1 failed"
    else
        cp $outdir/yolov4_416_coco_batch$1/compilation.bmodel $outdir/yolov4_416_coco_int8_$1b.bmodel
    fi

}

pushd $script_dir
# 1b
gen_int8bmodel 1
# 4b
gen_int8bmodel 4
popd
