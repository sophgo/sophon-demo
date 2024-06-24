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

function gen_mlir()
{
   model_transform.py \
        --model_name yolov9c \
        --model_def ../models/onnx/yolov9c_$1b.onnx \
        --input_shapes [[$1,3,640,640]] \
        --mean 0.0,0.0,0.0 \
        --scale 0.0039216,0.0039216,0.0039216 \
        --keep_aspect_ratio \
        --pixel_format rgb  \
        --test_input ../datasets/test/dog.jpg \
        --test_result yolov9c_top_outputs.npz \
        --mlir yolov9c_$1b.mlir
}

function gen_cali_table()
{
    run_calibration.py yolov9c_$1b.mlir \
        --dataset ../datasets/coco128/ \
        --input_num 32 \
        -o yolov9c_cali_table
}

function gen_int8bmodel()
{
    model_deploy.py \
        --mlir yolov9c_$1b.mlir \
        --quantize INT8 \
        --chip $target \
        --quantize_table ../models/onnx/yolov9c_${target}_qtable \
        --calibration_table yolov9c_cali_table \
        --test_input yolov9c_in_f32.npz \
        --test_reference yolov9c_top_outputs.npz \
        --model yolov9c_int8_$1b.bmodel
        # --tolerance 0.99,0.99 \ #手动搜敏感层可以把这两行注释取消掉，注意上一行最后要加一个 \，这样model_deploy就会打出哪些层比对不过。
        # --compare_all

    mv yolov9c_int8_$1b.bmodel $outdir/
    if test $target = "bm1688";then
        model_deploy.py \
            --mlir yolov9c_$1b.mlir \
            --quantize INT8 \
            --chip $target \
            --model yolov9c_int8_$1b_2core.bmodel \
            --quantize_table ../models/onnx/yolov9c_${target}_qtable \
            --calibration_table yolov9c_cali_table \
            --num_core 2 \
            --test_input yolov9c_in_f32.npz \
            --test_reference yolov9c_top_outputs.npz

        mv yolov9c_int8_$1b_2core.bmodel $outdir/
    fi
}


pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
gen_mlir 1
gen_cali_table 1
gen_int8bmodel 1

# batch_size=4
gen_mlir 4
gen_int8bmodel 4

popd
