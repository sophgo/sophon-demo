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

function gen_fp16bmodel()
{
    gen_mlir $1
    # Note: F16 with an F32-only mixed-precision qtable.
    # The box-decode head and the fused yolo_seg postprocess
    # (argmax/gather/compare/nms) must run in F32, otherwise
    # NMS dedup breaks on BM1688 and ~4x too many boxes are kept.
    # The F16 qtable has no INT8 entries (that would crash tpuc-opt
    # in F16 mode with a CalibratedQuantizedType assertion).
    #
    model_deploy.py \
        --mlir ${model_name}_seg_fuse_$1b.mlir \
        --quantize F16 \
        --chip  $target \
        --processor  $target \
        --fuse_preprocess \
        --customization_format BGR_PACKED \
        --quantize_table ${model_name}_seg_fuse_qtable_f16 \
        --model ${model_name}_seg_fuse_fp16_$1b.bmodel \
        --quant_output

    mv ${model_name}_seg_fuse_fp16_$1b.bmodel $outdir/
    if test $target = "bm1688";then
        model_deploy.py \
            --mlir ${model_name}_seg_fuse_$1b.mlir \
            --quantize F16 \
            --chip  $target \
            --processor  $target \
            --fuse_preprocess \
            --customization_format BGR_PACKED \
            --quantize_table ${model_name}_seg_fuse_qtable_f16 \
            --num_core 2 \
            --model ${model_name}_seg_fuse_fp16_$1b_2core.bmodel \
            --quant_output

        mv ${model_name}_seg_fuse_fp16_$1b_2core.bmodel $outdir/
    fi
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
model_name=yolov8s
gen_fp16bmodel 1

popd