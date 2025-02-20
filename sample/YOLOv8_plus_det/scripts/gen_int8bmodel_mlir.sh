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
    onnx_path=""
    if test $model_name = "yolov8s"; then
        onnx_path=../models/onnx/yolov8s.onnx
    elif test $model_name = "yolov9s"; then
        onnx_path=../models/onnx/yolov9-s-converted.onnx
    elif test $model_name = "yolov11s"; then
        onnx_path=../models/onnx/yolo11s.onnx
    elif test $model_name = "yolov12s"; then
        onnx_path=../models/onnx/yolov12s.onnx
    fi
    model_transform.py \
        --model_name ${model_name} \
        --model_def $onnx_path \
        --input_shapes [[$1,3,640,640]] \
        --mean 0.0,0.0,0.0 \
        --scale 0.0039216,0.0039216,0.0039216 \
        --keep_aspect_ratio \
        --pixel_format rgb \
        --mlir ${model_name}_$1b.mlir
        # --test_input ../datasets/test/dog.jpg \
        # --test_result ${model_name}_top_outputs.npz
}

function gen_cali_table()
{
    run_calibration.py ${model_name}_$1b.mlir \
        --dataset ../datasets/coco128/ \
        --input_num 128 \
        --inference_num 10 \
        -o ${model_name}_cali_table
        # --search search_qtable \
        # --quantize_table sensitive_layer.txt \
}

function gen_int8bmodel()
{
    model_deploy.py \
        --mlir ${model_name}_$1b.mlir \
        --quantize INT8 \
        --chip $target \
        --calibration_table ${model_name}_cali_table \
        --quantize_table ../models/onnx/${model_name}_qtable \
        --model ${model_name}_int8_$1b.bmodel
        # --test_input ../datasets/test/dog.jpg \
        # --test_reference ${model_name}_top_outputs.npz \
        # --tolerance 0.99,0.99 \
        # --compare_all

    mv ${model_name}_int8_$1b.bmodel $outdir/
    if test $target = "bm1688";then
        model_deploy.py \
            --mlir ${model_name}_$1b.mlir \
            --quantize INT8 \
            --chip $target \
            --model ${model_name}_int8_$1b_2core.bmodel \
            --calibration_table ${model_name}_cali_table \
            --quantize_table ../models/onnx/${model_name}_qtable \
            --num_core 2
            # --test_input ${model_name}_in_f32.npz \
            # --test_reference ${model_name}_top_outputs.npz \
            # --compare_all

        mv ${model_name}_int8_$1b_2core.bmodel $outdir/
    fi
}


pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
model_name=yolov8s
gen_mlir 1
gen_cali_table 1
gen_int8bmodel 1

# batch_size=4
gen_mlir 4
gen_int8bmodel 4

# batch_size=1
model_name=yolov9s
gen_mlir 1
gen_cali_table 1
gen_int8bmodel 1

# batch_size=4
gen_mlir 4
gen_int8bmodel 4

# batch_size=1
model_name=yolov11s
gen_mlir 1
gen_cali_table 1
gen_int8bmodel 1

# batch_size=4
gen_mlir 4
gen_int8bmodel 4

# batch_size=1
model_name=yolov12s
gen_mlir 1
gen_cali_table 1
gen_int8bmodel 1

batch_size=4
gen_mlir 4
gen_int8bmodel 4
popd
