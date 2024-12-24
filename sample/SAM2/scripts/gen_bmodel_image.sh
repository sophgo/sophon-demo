#!/bin/bash
# set -ex
model_dir=$(dirname $(readlink -f "$0"))

echo $model_dir

target=bm1684x
target_dir=BM1684X
mode=f16
batch_size=1

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

outdir=../models/$target_dir

function gen_mlir_image_encoder()
{
    model_transform.py \
        --model_name sam2_encoder \
        --model_def ../models/onnx/image/sam2_hiera_tiny_encoder.onnx \
        --input_shapes [[$1,3,1024,1024]] \
        --mlir sam2_encoder.mlir
}

function gen_bmodel_image_encoder()
{
    model_deploy.py \
        --mlir sam2_encoder.mlir \
        $quantize_args \
        --chip $target \
        --model sam2_encoder_${mode}_${1}b.bmodel

    mv sam2_encoder_${mode}_${1}b.bmodel $outdir/image_encoder/
}

function gen_bmodel_image_encoder_1688()
{
    model_deploy.py \
        --mlir sam2_encoder.mlir \
        $quantize_args \
        --chip $target \
        --num_core $2 \
        --model sam2_encoder_${mode}_${1}b_${2}core.bmodel

    mv sam2_encoder_${mode}_${1}b_${2}core.bmodel $outdir/image_encoder/
}

function gen_mlir_image_decoder()
{
    model_transform.py \
        --model_name sam2_decoder \
        --model_def ../models/onnx/image/sam2_hiera_tiny_decoder.onnx \
        --input_shapes [[$1,256,64,64],[$1,32,256,256],[$1,64,128,128],[$1,1,2],[$1,1],[$1,1,256,256],[$1]] \
        --mlir sam2_decoder.mlir
}

function gen_bmodel_image_decoder()
{
    model_deploy.py \
        --mlir sam2_decoder.mlir \
        $quantize_args \
        --chip $target \
        --model sam2_decoder_${mode}_${1}b.bmodel

    mv sam2_decoder_${mode}_${1}b.bmodel $outdir/image_decoder/
}

function gen_bmodel_image_decoder_1688()
{
    model_deploy.py \
        --mlir sam2_decoder.mlir \
        $quantize_args \
        --chip $target \
        --num_core $2 \
        --model sam2_decoder_${mode}_${1}b_${2}core.bmodel

    mv sam2_decoder_${mode}_${1}b_${2}core.bmodel $outdir/image_decoder/
}

pushd $model_dir
if [ ! -d $outdir/image_encoder ] ; then
    mkdir -p $outdir/image_encoder

else
    echo "Models folder exist! "
fi

if [ ! -d $outdir/image_decoder ] ; then
    mkdir -p $outdir/image_decoder

else
    echo "Models folder exist! "
fi

if [ x$target == x"bm1684x" ]; then
    gen_mlir_image_encoder $batch_size
    gen_bmodel_image_encoder $batch_size
    gen_mlir_image_decoder $batch_size
    gen_bmodel_image_decoder $batch_size
elif [ x$target == x"bm1688" ]; then
    gen_mlir_image_encoder $batch_size
    gen_bmodel_image_encoder_1688 $batch_size 1
    gen_bmodel_image_encoder_1688 $batch_size 2
    gen_mlir_image_decoder $batch_size
    gen_bmodel_image_decoder_1688 $batch_size 1
    gen_bmodel_image_decoder_1688 $batch_size 2
else
    echo "Error, unsupported chip"
    exit 1
fi

rm -f *.npz

popd
