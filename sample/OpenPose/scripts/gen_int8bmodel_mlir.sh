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

function gen_mlir_coco()
{   
    model_transform.py \
        --model_name pose_coco \
        --model_def ../models/caffe/pose/coco/pose_deploy_linevec.prototxt \
        --model_data ../models/caffe/pose/coco/pose_iter_440000.caffemodel \
        --input_shapes [[$1,3,368,368]] \
        --mean 128.0,128.0,128.0 \
        --scale 0.0039216,0.0039216,0.0039216 \
        --test_input ../datasets/test/1.jpg \
        --test_result pose_coco_top_outputs.npz \
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
        --chip $target \
        --calibration_table pose_coco_cali_table \
        --quantize_table ../models/caffe/pose_coco_qtable \
        --model pose_coco_int8_$1b.bmodel \
        --test_input pose_coco_in_f32.npz \
        --test_reference pose_coco_top_outputs.npz \
        --debug

    mv pose_coco_int8_$1b.bmodel $outdir/
    if test $target = "bm1688";then
        model_deploy.py \
            --mlir pose_coco_$1b.mlir \
            --quantize INT8 \
            --chip $target \
            --model pose_coco_int8_$1b_2core.bmodel \
            --calibration_table pose_coco_cali_table \
            --quantize_table ../models/caffe/pose_coco_qtable \
            --num_core 2 \
            --test_input pose_coco_in_f32.npz \
            --test_reference pose_coco_top_outputs.npz \
            --debug 
        mv pose_coco_int8_$1b_2core.bmodel $outdir/
    fi
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