#!/bin/bash
# set -ex
model_dir=$(dirname $(readlink -f "$0"))

echo $model_dir

target=bm1684x
target_dir=BM1684X
mode=f16
# TO DO
batch_size=1
num_core=1

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --chip)
            target=${2,,}
            target_dir=${target^^}
            shift 2
            ;;
        --mode)
            mode="${2}"
            quantize_args="--quantize ${mode^^}"
            shift 2
            ;;
        *)
            echo "Invalid option: $key" >&2
            exit 1
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            exit 1
            ;;
    esac
done

if [[ x"$target" != "bm1688" ]]; then
    num_core=1
fi

outdir=../models/$target_dir/video/

function gen_mlir_image_encoder()
{
    model_transform.py \
        --model_name sam2_video_image_encoder \
        --model_def ../models/onnx/video/image_encoder_no_pos.onnx\
        --input_shapes [[1,3,1024,1024]] \
        --mlir sam2_image_encoder_no_pos.mlir
}

function gen_bmodel_image_encoder()
{
    model_deploy.py \
        --mlir sam2_image_encoder_no_pos.mlir \
        $quantize_args \
        --chip $target \
        --num_core $num_core \
        --model sam2_image_encoder_no_pos.bmodel

    mv sam2_image_encoder_no_pos.bmodel $outdir
}

function gen_mlir_image_decoder()
{
    model_transform.py \
        --model_name sam2_image_decoder \
        --model_def ../models/onnx/video/image_decoder.onnx \
        --input_shapes [[1,2,2],[1,2],[1,256,64,64],[1,32,256,256],[1,64,128,128]] \
        --mlir sam2_image_decoder.mlir
}

function gen_bmodel_image_decoder()
{
    model_deploy.py \
        --mlir sam2_image_decoder.mlir \
        $quantize_args \
        --chip $target \
        --num_core $num_core \
        --model sam2_image_decoder.bmodel

    mv sam2_image_decoder.bmodel $outdir
}

function gen_mlir_memory_attention()
{
    model_transform.py \
        --model_name sam2_memory_attention \
        --model_def ../models/onnx/video/memory_attention_nomatmul.onnx \
        --input_shapes [[1,256,64,64],[4096,1,256],[28672,1,64],[64,1,64],[28736,1,64]] \
        --mlir sam2_memory_attention_nomatmul.mlir
}

function gen_bmodel_memory_attention()
{
    model_deploy.py \
        --mlir sam2_memory_attention_nomatmul.mlir \
        --quantize F16 \
        --chip $target \
        --num_core $num_core \
        --model sam2_memory_attention_nomatmul.bmodel

    mv sam2_memory_attention_nomatmul.bmodel $outdir
}

function gen_mlir_memory_encoder()
{
    model_transform.py \
        --model_name sam2_memory_encoder \
        --model_def ../models/onnx/video/memory_encoder.onnx \
        --input_shapes [[1,1,1024,1024],[1,256,64,64]] \
        --mlir sam2_memory_encoder.mlir
}

function gen_bmodel_memory_encoder()
{
    model_deploy.py \
        --mlir sam2_memory_encoder.mlir \
        --quantize F16 \
        --chip $target \
        --num_core $num_core \
        --model sam2_memory_encoder.bmodel

    mv sam2_memory_encoder.bmodel $outdir
}

pushd $model_dir

if [ ! -d $outdir ] ; then
    mkdir -p $outdir

else
    echo "Models folder exist! "
fi

gen_mlir_image_encoder
gen_bmodel_image_encoder

gen_mlir_image_decoder
gen_bmodel_image_decoder

gen_mlir_memory_attention
gen_bmodel_memory_attention

gen_mlir_memory_encoder
gen_bmodel_memory_encoder

popd