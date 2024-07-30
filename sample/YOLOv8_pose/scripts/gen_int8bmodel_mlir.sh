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

fp_forward_chip=$target
if test $target = "bm1688"; then
    fp_forward_chip=bm1684x
fi

function gen_mlir()
{
   model_transform.py \
        --model_name yolov8s-pose${opt} \
        --model_def ../models/onnx/yolov8s-pose${opt}.onnx \
        --input_shapes [[$1,3,640,640]] \
        --mean 0.0,0.0,0.0 \
        --scale 0.0039216,0.0039216,0.0039216 \
        --keep_aspect_ratio \
        --pixel_format rgb  \
        --test_input ../datasets/test/1.jpg \
        --test_result yolov8s-pose${opt}_top_outputs.npz \
        --mlir yolov8s-pose${opt}_$1b.mlir
}

function gen_cali_table()
{
    run_calibration.py yolov8s-pose${opt}_$1b.mlir \
        --dataset ../datasets/coco128/ \
        --input_num 32 \
        -o yolov8s-pose${opt}_cali_table
}

function gen_qtable()
{
    fp_forward.py yolov8s-pose${opt}_$1b.mlir \
        --quantize INT8 \
        --chip $fp_forward_chip \
        --fpfwd_outputs /model.22/dfl/conv/Conv_output_0_Conv,/model.22/cv2.2/cv2.2.2/Conv_output_0_Conv,/model.22/cv3.2/cv3.2.2/Conv_output_0_Conv,/model.22/cv4.0/cv4.0.2/Conv_output_0_Conv,/model.22/cv4.1/cv4.1.2/Conv_output_0_Conv,/model.22/cv4.2/cv4.2.2/Conv_output_0_Conv \
        -o yolov8-pose${opt}_qtable
}

function gen_int8bmodel()
{
    qtable_path=../models/onnx/yolov8s-pose${opt}_qtable_fp16
    if test $target = "bm1684";then
        qtable_path=../models/onnx/yolov8s-pose${opt}_qtable_fp32
    fi
    model_deploy.py \
        --mlir yolov8s-pose${opt}_$1b.mlir \
        --quantize INT8 \
        --chip $target \
        --calibration_table yolov8s-pose${opt}_cali_table \
        --quantize_table yolov8-pose${opt}_qtable \
        --test_input yolov8s-pose${opt}_in_f32.npz \
        --test_reference yolov8s-pose${opt}_top_outputs.npz \
        --model yolov8s-pose${opt}_int8_$1b.bmodel

    mv yolov8s-pose${opt}_int8_$1b.bmodel $outdir/
    if test $target = "bm1688";then
        model_deploy.py \
            --mlir yolov8s-pose${opt}_$1b.mlir \
            --quantize INT8 \
            --chip $target \
            --model yolov8s-pose${opt}_int8_$1b_2core.bmodel \
            --calibration_table yolov8s-pose${opt}_cali_table \
            --num_core 2 \
            --quantize_table yolov8-pose${opt}_qtable \
            --test_input yolov8s-pose${opt}_in_f32.npz \
            --test_reference yolov8s-pose${opt}_top_outputs.npz 


        mv yolov8s-pose${opt}_int8_$1b_2core.bmodel $outdir/
    fi
}


pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
gen_mlir 1
gen_cali_table 1
gen_qtable 1
gen_int8bmodel 1

# batch_size=4
gen_mlir 4
gen_int8bmodel 4

# opt="_opt"
# # batch_size=1
# gen_mlir 1
# gen_cali_table 1
# gen_int8bmodel 1

# # batch_size=4
# gen_mlir 4
# gen_int8bmodel 4

popd