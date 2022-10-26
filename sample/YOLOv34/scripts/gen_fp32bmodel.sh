#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))
target=$1
outdir=../data/models/$target
 
 
function gen_fp32bmodel()
{
    bmnetd --net_name=yolov4_416_coco \
	              --weight=../data/models/darknet/yolov4.weights \
                      --target=$target \
                      --opt=2 \
                      --cmp=true \
                      --shapes=[$1,3,416,416] \
                      --model=../data/models/darknet/yolov4.cfg \
                      --outdir=$outdir \
                      --dyn=false 
    mv $outdir/compilation.bmodel $outdir/yolov4_416_coco_fp32_$1b.bmodel
 
}
 
pushd $model_dir
#batch_size=1
gen_fp32bmodel 1
#batch_size=4
gen_fp32bmodel 4
popd

pushd $outdir
rm -f *dat
popd



