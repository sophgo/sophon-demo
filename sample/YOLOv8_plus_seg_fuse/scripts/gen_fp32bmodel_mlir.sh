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

function gen_fp32bmodel()
{
    gen_mlir $1
    model_deploy.py \
        --mlir ${model_name}_seg_fuse_$1b.mlir \
        --quantize F32 \
        --chip  $target \
        --processor  $target \
        --fuse_preprocess \
        --customization_format BGR_PACKED \
        --model ${model_name}_seg_fuse_fp32_$1b.bmodel \
        --quant_output

    mv ${model_name}_seg_fuse_fp32_$1b.bmodel $outdir/
    if test $target = "bm1688";then
        model_deploy.py \
            --mlir ${model_name}_seg_fuse_$1b.mlir \
            --quantize F32 \
            --chip  $target \
            --processor  $target \
            --fuse_preprocess \
            --customization_format BGR_PACKED \
            --num_core 2 \
            --model ${model_name}_seg_fuse_fp32_$1b_2core.bmodel \
            --quant_output

        mv ${model_name}_seg_fuse_fp32_$1b_2core.bmodel $outdir/
    fi
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
model_name=yolov8s
gen_fp32bmodel 1

popd