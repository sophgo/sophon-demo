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

if [ ! $2 ]; then
    model_type=SAM-ViT-B
else
    model_type=$2
fi

if test $model_type = "SAM-ViT-B"; then
    onnx_path="../models/onnx/vit-b-auto-multi_mask.onnx"
elif test $model_type = "SAM-ViT-T"; then
    onnx_path="../models/onnx/vit-t-auto-multi_mask.onnx"
else
    echo "unsupport model_type: $model_type, only supports: SAM-ViT-B,SAM-ViT-T."
fi

function gen_mlir_decoder()
{
    model_transform.py \
        --model_name auto-sam_decoder \
        --model_def $onnx_path \
        --input_shapes [[$1,256,64,64],[64,1,2],[64,1],[64,1,256,256],[1],[2]] \
        --output_names /Concat_3_output_0,/Slice_2_output_0,iou_predictions,low_res_masks \
        --mlir sam_auto_decoder_$1b.mlir
}

function gen_fp32bmodel_decoder()
{
    model_deploy.py \
        --mlir sam_auto_decoder_$1b.mlir \
        --quantize F32 \
        --chip $target \
	    --model ${model_type}_auto_multi_decoder_fp32_$1b.bmodel

    mv ${model_type}_auto_multi_decoder_fp32_$1b.bmodel $outdir/decode_bmodel
}


pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir/decode_bmodel
else
    echo "Models folder exist! "
fi

# batch_size=1
gen_mlir_decoder 1
gen_fp32bmodel_decoder 1

popd
