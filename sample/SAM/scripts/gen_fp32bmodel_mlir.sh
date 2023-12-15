#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))

target=bm1684x
target_dir=BM1684X


outdir=../models/$target_dir

function gen_mlir_decoder()
{
    model_transform.py \
        --model_name sam_decoder \
        --model_def ../models/onnx/vit-b-scripts.onnx \
        --input_shapes [[$1,256,64,64],[1,5,2],[1,5],[1,1,256,256],[1],[2]] \
        --output_names /Concat_18_output_0,/Slice_9_output_0,iou_predictions,low_res_masks \
        --mlir sam_decoder_$1b.mlir
}

function gen_fp32bmodel_decoder()
{
    model_deploy.py \
        --mlir sam_decoder_$1b.mlir \
        --quantize F16 \
        --chip $target \
        --model SAM-ViT-B_decoder_fp32_$1b.bmodel

    mv SAM-ViT-B_decoder_fp32_$1b.bmodel $outdir/
}


pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir/decode_bmodel
fi

# batch_size=1
gen_mlir_decoder 1
gen_fp32bmodel_decoder 1

popd
