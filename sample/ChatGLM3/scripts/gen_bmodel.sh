#!/bin/bash
# set -ex
model_dir=$(dirname $(readlink -f "$0"))
pushd $model_dir

models=
mode="fp16"
num_device=1
num_core=1
quantize_args="--quantize F16"
addr_args=""
quantio_args=""
quant_io=1
device_args=""
target="bm1684x"
target_dir="BM1684X"
out_model_name=chatglm3-6b
num_layers=27

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --mode)
            mode="$2"
            shift 2
            ;;
        --num_device)
            num_device="$2"
            shift 2
            ;;
        --num_core)
            num_core="$2"
            shift 2
            ;;
        --target)
            target=${2,,}
            target_dir=${target^^}
            shift 2
            ;;
        --addr_mode)
            addr_mode="$2"
            shift 2
            ;;
        --no_quant_io)
            quant_io=0
            shift 1
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

if [ $num_core != 1 ] && [ x$target != x"bm1688" ]; then
    echo "Failed: only bm1688 support 2core bmodels."
    exit 1
fi

out_model_name="${out_model_name}_${mode}"

if [ x$mode == x"int8" ]; then
    quantize_args="--quantize W8F16"
elif [ x$mode == x"bf16" ] || [ x$mode == x"fp16" ]; then
    quantize_args="--quantize F16"
elif [ x$mode == x"int4" ]; then
    quantize_args="--quantize W4F16 --q_group_size 64"
else
    echo "Error, unknown quantize mode, only support fp16/bf16/int8/int4"
    exit 1
fi

if [ $quant_io == 1 ]; then
    quantize_args="$quantize_args --quant_input --quant_output"
    quantio_args="--quant_input --quant_output"
fi

if [ x$num_device != x1 ]; then
    device_args="--num_device $num_device"
    out_model_name="${out_model_name}_${num_device}dev"
fi

out_model="${out_model_name}_${num_core}core.bmodel"

if [ x$addr_mode == x"io_alone" ]; then
    addr_args="--addr_mode io_alone"
    out_model="${out_model_name}_${num_core}core_io_alone.bmodel"
fi


outdir=tmp/embedding
mkdir -p $outdir
pushd $outdir

model_transform.py \
    --model_name embedding \
    --model_def $model_dir/../models/onnx/embedding.onnx \
    --mlir embedding.mlir


model_deploy.py \
    --mlir embedding.mlir \
    --quantize F16 \
    --chip $target \
    $quantio_args \
    --num_core $num_core \
    --model embedding.bmodel

model_transform.py \
    --model_name embedding_cache \
    --model_def $model_dir/../models/onnx/embedding.onnx \
    --input_shapes [[1,1]] \
    --mlir embedding_cache.mlir


model_deploy.py \
    --mlir embedding_cache.mlir \
    --quantize F16 \
    --chip $target \
    $quantio_args \
    --num_core $num_core \
    --model embedding_cache.bmodel

rm *.npz

models=$models' '$outdir'/embedding.bmodel '$outdir'/embedding_cache.bmodel '

popd

echo $models

outdir=tmp/$mode"_"$num_device"dev"/lm_head
mkdir -p $outdir
pushd $outdir

model_transform.py \
    --model_name lm_head \
    --model_def $model_dir/../models/onnx/lm_head.onnx \
    --mlir lm_head.mlir

model_deploy.py \
    --mlir lm_head.mlir \
    --quantize F16 \
    --chip $target \
    $device_args \
    $quantio_args \
    --num_core $num_core \
    --model lm_head.bmodel

models=${models}${outdir}'/lm_head.bmodel '
popd

echo $models

outdir=tmp/$mode"_"$num_device"dev"/block
mkdir -p $outdir

pushd $outdir
mkdir -p $outdir

for ((i=0; i<=$num_layers; i++)); do

    model_transform.py \
        --model_name block_$i \
        --model_def $model_dir/../models/onnx/block_$i.onnx \
        --mlir block_$i.mlir

    model_deploy.py \
        --mlir block_$i.mlir \
        $quantize_args \
        --chip $target \
        $device_args \
        --num_core $num_core \
        --model block_$i.bmodel

    model_transform.py \
        --model_name block_cache_$i \
        --model_def $model_dir/../models/onnx/block_cache_$i.onnx \
        --mlir block_cache_$i.mlir

    model_deploy.py \
        --mlir block_cache_$i.mlir \
        $quantize_args \
        --chip $target \
        $device_args \
        --num_core $num_core \
         $addr_args \
        --model block_cache_$i.bmodel
        
    rm *.npz

    models=${models}${outdir}'/block_'$i'.bmodel '$outdir'/block_cache_'$i'.bmodel '

done
popd
echo $models

outdir=../models/$target_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
model_tool --combine $models -o $outdir/$out_model
chmod 777 $outdir/$out_model

popd #model_dir