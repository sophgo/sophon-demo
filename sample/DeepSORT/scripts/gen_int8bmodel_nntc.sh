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
            --net_name=extractor  \
            --model=../models/torch/extractor.pt  \
            --cali_image_path=../datasets/cali_set  \
            --cali_iterations=128   \
            --cali_image_preprocess='resize_h=128,resize_w=64;
                            mean_value=2.1179039:1.9912664:1.772926,
                            scale=0.0171248:0.017507:0.0174292,bgr2rgb=True'   \
            --input_shapes="[1,3,128,64]"  \
            --target=$target   \
            --convert_bmodel_cmd_opt="-opt=1"   \
            --try_cali_accuracy_opt="-th_method=ADMM" #-fpfwd_outputs=< 24 >86,< 24 >55,< 24 >18;
    mv ../models/torch/extractor_batch1/compilation.bmodel $outdir/extractor_int8_1b.bmodel
}

function gen_int8bmodel()
{
    bmnetu --model=../models/torch/extractor_bmnetp_deploy_int8_unique_top.prototxt  \
           --weight=../models/torch/extractor_bmnetp.int8umodel \
           --net_name=extractor \
           --shapes=[$1,3,128,64] \
           --target=$target \
           --opt=1
    mv compilation/compilation.bmodel $outdir/extractor_int8_$1b.bmodel
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