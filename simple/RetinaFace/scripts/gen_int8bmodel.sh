#!/bin/bash
script_dir=$(dirname $(readlink -f "$0"))

if [ ! $1 ]; then
    target="BM1684X"
else
    target=$1
fi

outdir=../data/models/$target
cali_data_path=../data/images/cali_dataset


function gen_int8bmodel()
{

    python3 -m ufw.cali.cali_model \
        --net_name 'retinaface_mobilenet0.25' \
        --model ../data/models/onnx/retinaface_mobilenet0.25.onnx \
        --cali_image_path $cali_data_path \
        --cali_image_preprocess='resize_h=640,resize_w=640;mean_value=104:117:123,scale=1.0,bgr2rgb=False' \
        --input_shapes [$1,3,640,640] \
        --convert_bmodel_cmd_opt "-outdir $outdir --target $target -input_as_fp32=input"\
        --try_cali_accuracy_opt='-fpfwd_outputs=449,463,477'

    if [ $? -ne 0 ]; then
        echo "gen_int8bmodel batch_size $1 failed"
    else
        mv $outdir/compilation.bmodel $outdir/retinaface_mobilenet0.25_int8_$1b.bmodel
    fi
}

pushd $script_dir
# b1
gen_int8bmodel 1
# b4
gen_int8bmodel 4
popd
