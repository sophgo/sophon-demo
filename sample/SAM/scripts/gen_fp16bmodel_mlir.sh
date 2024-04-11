#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))

target=bm1684x
target_dir=BM1684X


outdir=../models/$target_dir


function gen_mlir_embedding()
{
    model_transform.py \
        --model_name sam_embedding \
        --model_def ../models/onnx/embedding_model.onnx \
        --input_shapes [[1,3,1024,1024]] \
        --mlir sam_embedding_$1b.mlir
}

function gen_fp16bmodel_embedding()
{
    model_deploy.py \
        --mlir sam_embedding_$1b.mlir \
        --quantize F16 \
        --chip $target \
        --model SAM-ViT-B_embedding_fp16_$1b.bmodel

    mv SAM-ViT-B_embedding_fp16_$1b.bmodel $outdir/embedding_bmodel/
}


function gen_mlir_single_decoder()
{
    model_transform.py \
        --model_name sam_decoder \
        --model_def ../models/onnx/decode_model_single_mask.onnx \
        --input_shapes [[$1,256,64,64],[1,2,2],[1,2],[1,1,256,256],[1],[2]] \
        --output_names /Concat_18_output_0,/Slice_9_output_0,iou_predictions,low_res_masks \
        --mlir sam_decoder_single_mask_$1b.mlir
}

function gen_fp16bmodel_single_decoder()
{
    model_deploy.py \
        --mlir sam_decoder_single_mask_$1b.mlir \
        --quantize F16 \
        --chip $target \
        --model SAM-ViT-B_decoder_single_mask_fp16_$1b.bmodel

    mv SAM-ViT-B_decoder_single_mask_fp16_$1b.bmodel $outdir/decode_bmodel/
}


function gen_mlir_multi_decoder()
{
    model_transform.py \
        --model_name sam_decoder \
        --model_def ../models/onnx/decode_model_multi_mask.onnx \
        --input_shapes [[$1,256,64,64],[1,2,2],[1,2],[1,1,256,256],[1],[2]] \
        --output_names /Concat_15_output_0,/Slice_9_output_0,iou_predictions,low_res_masks \
        --mlir sam_decoder_multi_mask_$1b.mlir
}

function gen_fp16bmodel_multi_decoder()
{
    model_deploy.py \
        --mlir sam_decoder_multi_mask_$1b.mlir \
        --quantize F16 \
        --chip $target \
        --model SAM-ViT-B_decoder_multi_mask_fp16_$1b.bmodel

    mv SAM-ViT-B_decoder_multi_mask_fp16_$1b.bmodel $outdir/decode_bmodel/
}


pushd $model_dir
if [ ! -d $outdir/embedding_bmodel ] ; then
    mkdir -p $outdir/embedding_bmodel

else
    echo "Models folder exist! "
fi

if [ ! -d $outdir/decode_bmodel ] ; then
    mkdir -p $outdir/decode_bmodel

else
    echo "Models folder exist! "
fi

# batch_size=1
gen_mlir_embedding 1
gen_fp16bmodel_embedding 1

gen_mlir_single_decoder 1
gen_fp16bmodel_single_decoder 1

gen_mlir_multi_decoder 1
gen_fp16bmodel_multi_decoder 1
popd
