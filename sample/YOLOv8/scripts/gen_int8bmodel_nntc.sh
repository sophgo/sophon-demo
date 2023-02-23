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
                --net_name=yolov8s  \
                --model=../models/torch/yolov8s.torchscript.pt  \
                --cali_image_path=../datasets/coco128  \
                --cali_iterations=128   \
                --cali_image_preprocess='resize_h=640,resize_w=640;scale=0.003921569,bgr2rgb=True'   \
                --input_shapes="[1,3,640,640]"  \
                --target=$target   \
                --convert_bmodel_cmd_opt="-opt=1"   \
                --test_iterations 30 \
                --try_cali_accuracy_opt="-fpfwd_outputs=< 22 >93,< 22 >x.1;-th_method=MSE;-per_channel=True"

    mv ../models/torch/yolov8s_batch1/compilation.bmodel $outdir/yolov8s_int8_1b.bmodel
}

function gen_int8bmodel()
{
    bmnetu --model=../models/torch/yolov8s_bmnetp_deploy_int8_unique_top.prototxt  \
           --weight=../models/torch/yolov8s_bmnetp.int8umodel \
           -net_name=yolov8s \
           --shapes=[$1,3,640,640] \
           -target=$target \
           -opt=1
    mv compilation/compilation.bmodel $outdir/yolov8s_int8_$1b.bmodel
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