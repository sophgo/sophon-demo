#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))


function gen_mlir()
{
    model_transform \
    --model_name yolov5s \
    --model_def ../models/onnx/yolov5s_v6.1_3output.onnx \
    --input_shapes [[$1,3,640,640]] \
    --output_names 366,326,346  \
    --add_postprocess yolov5 \
    --pixel_format rgb \
    --scale 0.0039216,0.0039216,0.0039216 \
    --mean 0.0,0.0,0.0 \
    --keep_aspect_ratio \
    --mlir yolov5s_v6.1_3output_$1b.mlir
}

pushd $model_dir
gen_mlir 1
gen_mlir 4
popd