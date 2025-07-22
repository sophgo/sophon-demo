#!/bin/bash
scripts_dir=$(dirname $(readlink -f "$0"))
pushd $scripts_dir

model_dir=$(dirname $(readlink -f "$0"))
mode=fp16
quantize_args="--quantize F16"
chip=bm1684x
onnx_dir=../tools/tmp/onnx_tts
while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
    --chip)
        chip="$2"
        shift 2
        ;;
    --mode)
        mode="$2"
        shift 2
        ;;
    --onnx_dir)
        onnx_dir=$scripts_dir/$2
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

if [ x$mode == x"fp16" ]; then
    quantize_args="--quantize F16"
elif [ x$mode == x"fp32" ]; then
    quantize_args="--quantize F32"
else
    echo "Error, unknown quantize mode"
    exit 1
fi

outdir=tmp/vqvae
mkdir -p $outdir
pushd $outdir
function gen_mlir()
{
    model_transform.py \
        --model_name vqvae \
        --model_def $onnx_dir/vqvae.onnx \
        --input_shapes [[1,60,1],[1,1,8]] \
        --mlir vqvae.mlir
}

function gen_bmodel()
{
    model_deploy.py \
        --mlir vqvae.mlir \
        $quantize_args \
        --chip $chip \
        --model vqvae_${mode}_1b.bmodel
        
    mv vqvae_${mode}_1b.bmodel ../../
}

gen_mlir
gen_bmodel
popd


popd
