#!/bin/bash
root_dir=$(cd `dirname $BASH_SOURCE[0]`/ && pwd)/..

if [ ! $1 ]; then
    echo "Please set the target chip. Option: BM1684 and BM1684X"
    exit
else
    target=$1
fi

model_dir="${root_dir}/models/torch"
output_dir="${root_dir}/models/${target}"

echo outdir is $output_dir

function auto_cali()
{
    python3 -m ufw.cali.cali_model  \
            --net_name=yolov7  \
            --model=${model_dir}/yolov7_v0.1_3outputs.torchscript.pt  \
            --cali_image_path=${root_dir}/datasets/coco128  \
            --cali_iterations=128   \
            --cali_image_preprocess='resize_h=640,resize_w=640;scale=0.003921569,bgr2rgb=True'   \
            --input_shapes="[1,3,640,640]"  \
            --target=$target   \
            --convert_bmodel_cmd_opt="-opt=2"   \
            --try_cali_accuracy_opt="-fpfwd_outputs=< 105 >86,< 105 >55,< 105 >18;-th_method=MSE;-conv_group=true;-per_channel=true;-accuracy_opt=true;-graph_transform=true;-iterations=200;-dump_dist=./calibration_use_pb_dump_dist;-load_dist=./calibration_use_pb_dump_dist" \
            --postprocess_and_calc_score_class=feature_similarity
    cp ${model_dir}/yolov7_batch1/compilation.bmodel $output_dir/yolov7_v0.1_3output_int8_1b.bmodel
}

function gen_int8bmodel()
{
    bmnetu --model=${model_dir}/yolov7_bmnetp_deploy_int8_unique_top.prototxt  \
           --weight=${model_dir}/yolov7_bmnetp.int8umodel \
           -net_name=yolov7 \
           --shapes=[$1,3,640,640] \
           -target=$target \
           -opt=1
    cp compilation/compilation.bmodel $output_dir/yolov7_v0.1_3output_int8_$1b.bmodel
}

pushd $root_dir
if [ ! -d $output_dir ]; then
    mkdir -p $output_dir
fi
# batch_size=1
auto_cali
#gen_int8bmodel 1
# batch_size=4
gen_int8bmodel 4

popd