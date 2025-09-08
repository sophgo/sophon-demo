#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))
if [ ! $1 ]; then
    target=bm1684x
    target_dir=BM1684X
else
    target=${1,,}
    target_dir=${target^^}
fi

if [ ! $2 ]; then
    model_type=SAM-ViT-B
else
    model_type=$2
fi

if test $model_type = "SAM-ViT-B"; then
    decode_single_onnx="../models/onnx/decode_model_single_mask.onnx"
    decode_multi_onnx="../models/onnx/decode_model_multi_mask.onnx"
elif test $model_type = "SAM-ViT-T"; then
    decode_single_onnx="../models/onnx/decode_model_single_mask_mobile.onnx"
    decode_multi_onnx="../models/onnx/decode_model_multi_mask_mobile.onnx"
else
    echo "unsupport model_type: $model_type, only supports: SAM-ViT-B,SAM-ViT-T."
fi

outdir=../models/$target_dir

function gen_mlir_single_decoder()
{
    model_transform.py \
        --model_name sam_decoder \
        --model_def $decode_single_onnx \
        --input_shapes [[$1,256,64,64],[1,2,2],[1,2],[1,1,256,256],[1],[2]] \
        --output_names /Concat_18_output_0,/Slice_9_output_0,iou_predictions,low_res_masks \
        --mlir sam_decoder_single_mask_$1b.mlir
}

function gen_fp32bmodel_single_decoder()
{
    model_deploy.py \
        --mlir sam_decoder_single_mask_$1b.mlir \
        --quantize F32 \
        --chip $target \
        --model ${model_type}_decoder_single_mask_fp32_$1b.bmodel

    mv ${model_type}_decoder_single_mask_fp32_$1b.bmodel $outdir/decode_bmodel/
}


function gen_mlir_multi_decoder()
{
    model_transform.py \
        --model_name sam_decoder \
        --model_def $decode_multi_onnx \
        --input_shapes [[$1,256,64,64],[1,2,2],[1,2],[1,1,256,256],[1],[2]] \
        --output_names /Concat_15_output_0,/Slice_9_output_0,iou_predictions,low_res_masks \
        --mlir sam_decoder_multi_mask_$1b.mlir
}

function gen_fp32bmodel_multi_decoder()
{
    model_deploy.py \
        --mlir sam_decoder_multi_mask_$1b.mlir \
        --quantize F32 \
        --chip $target \
        --model ${model_type}_decoder_multi_mask_fp32_$1b.bmodel

    mv ${model_type}_decoder_multi_mask_fp32_$1b.bmodel $outdir/decode_bmodel/
}


pushd $model_dir
if [ ! -d $outdir/decode_bmodel ]; then
    mkdir -p $outdir/decode_bmodel
else
    echo "Models folder exist! "
fi
# batch_size=1
gen_mlir_single_decoder 1
gen_fp32bmodel_single_decoder 1

gen_mlir_multi_decoder 1
gen_fp32bmodel_multi_decoder 1

popd
