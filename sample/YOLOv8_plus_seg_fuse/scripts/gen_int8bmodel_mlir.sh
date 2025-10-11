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
    onnx_path=../models/onnx/yolov8s-seg.onnx
    if test $model_name = "yolov8s"; then
        onnx_path=../models/onnx/yolov8s-seg.onnx
    elif test $model_name = "yolov9c"; then
        onnx_path=../models/onnx/yolov9-c-seg-converted.onnx
    fi
    model_transform.py \
        --model_name $model_name \
        --model_def $onnx_path \
        --input_shapes [[$1,3,640,640]] \
        --add_postprocess yolov8_seg \
        --pixel_format rgb \
        --scale 0.0039216,0.0039216,0.0039216 \
        --mean 0.0,0.0,0.0 \
        --keep_aspect_ratio \
        --mlir ${model_name}_seg_fuse_$1b.mlir
}

function gen_cali_table()
{
    onnx_path=../models/onnx/yolov8s-seg.onnx
    if test $model_name = "yolov8s"; then
        onnx_path=../models/onnx/yolov8s-seg.onnx
    elif test $model_name = "yolov9c"; then
        onnx_path=../models/onnx/yolov9-c-seg-converted.onnx
    fi
    model_transform.py \
        --model_name $model_name \
        --model_def $onnx_path \
        --input_shapes [[$1,3,640,640]] \
        --pixel_format rgb \
        --scale 0.0039216,0.0039216,0.0039216 \
        --mean 0.0,0.0,0.0 \
        --keep_aspect_ratio \
        --mlir ${model_name}_seg_$1b.mlir
        
    run_calibration.py ${model_name}_seg_$1b.mlir \
        --dataset ../datasets/coco128/ \
        --input_num 16 \
        -o ${model_name}_seg_cali_table
}

function gen_int8bmodel()
{
    gen_mlir $1
    # fpfwd_outputs_layer_name='output1_Mul,output0_Concat,/model.22/dfl/conv/Conv_output_0_Conv'
    # fp_forward.py ${model_name}_seg_fuse_$1b.mlir --fpfwd_outputs $fpfwd_outputs_layer_name --chip $target --fp_type F32 -o ${model_name}_seg_fuse_qtable
    model_deploy.py \
        --mlir ${model_name}_seg_fuse_$1b.mlir \
        --quantize INT8 \
        --chip  $target \
        --processor  $target \
        --fuse_preprocess \
        --calibration_table ${model_name}_seg_cali_table \
        --quantize_table ${model_name}_seg_fuse_qtable \
        --customization_format BGR_PACKED \
        --model ${model_name}_seg_fuse_int8_$1b.bmodel \
        --quant_output

    mv ${model_name}_seg_fuse_int8_$1b.bmodel $outdir/
    if test $target = "bm1688";then
        model_deploy.py \
            --mlir ${model_name}_seg_fuse_$1b.mlir \
            --quantize INT8 \
            --chip  $target \
            --processor  $target \
            --fuse_preprocess \
            --calibration_table ${model_name}_seg_cali_table \
            --quantize_table ${model_name}_seg_fuse_qtable \
            --customization_format BGR_PACKED \
            --num_core 2 \
            --model ${model_name}_seg_fuse_int8_$1b_2core.bmodel \
            --quant_output

        mv ${model_name}_seg_fuse_int8_$1b_2core.bmodel $outdir/
    fi
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
model_name=yolov8s
gen_cali_table 1
gen_int8bmodel 1

popd