#!/bin/bash
set -ex
models=
mode="int8"
folder="./models/onnx"
num_device=1
mode_args=""
device_args=""
quantize_args="--quantize W8BF16"
name=""
num_layers=
out_model=$name.bmodel
addr_flag=false

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
    --name)
        name="$2"
        shift 2
        ;;
    --addr_flag)
        addr_flag=true
        shift
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

if [ "$name" = "qwen-1_8b" ]; then
  num_layers=23
  echo "Compile Qwen-1_8B"
elif [ "$name" = "qwen-7b" ]; then 
  num_layers=31
  echo "Compile Qwen-7B"
elif [ "$name" = "qwen-14b" ]; then
  num_layers=39
  echo "Compile Qwen-14B"
else
  >&2 echo -e "Error: Invalid name $name, the input name must be \033[31mqwen-1_8b|qwen-7b|qwen-14b\033[0m"
  exit 1
fi

if [ x$mode == x"int8" ]; then
    quantize_args="--quantize W8BF16"
elif [ x$mode == x"bf16" ]; then
    quantize_args="--quantize BF16"
elif [ x$mode == x"int4" ]; then
    quantize_args="--quantize W4BF16 --q_group_size 64"
else
    echo "Error, unknown quantize mode"
    exit 1
fi

if [ x$num_device != x1 ]; then
    device_args="--num_device $num_device"
    out_model=$name'_'$mode'_'$num_device'dev.bmodel'
else
    out_model=$name'_'$mode'_1dev.bmodel'
fi

addr_mode_flag=""
if [[ "$addr_flag" = true ]]; then
    addr_mode_flag="--addr_mode io_alone"
fi

outdir=${folder}/embedding
mkdir -p $outdir
pushd $outdir

model_transform.py \
    --model_name embedding \
    --model_def ../embedding.onnx \
    --mlir embedding.mlir

model_deploy.py \
    --mlir embedding.mlir \
    --quantize BF16 \
    --quant_input \
    --quant_output \
    --chip bm1684x \
    $device_args \
    --model embedding.bmodel

model_transform.py \
    --model_name embedding_cache \
    --model_def ../embedding.onnx \
    --input_shapes [[1,1]] \
    --mlir embedding_cache.mlir

model_deploy.py \
    --mlir embedding_cache.mlir \
    --quantize BF16 \
    --quant_input \
    --quant_output \
    --chip bm1684x \
    $device_args \
    --model embedding_cache.bmodel

rm *.npz

models=$models' '$outdir'/embedding.bmodel '$outdir'/embedding_cache.bmodel '

popd

echo $models

outdir=${folder}/$mode"_"$num_device"dev"/lm_head
mkdir -p $outdir
pushd $outdir

model_transform.py \
    --model_name lm_head \
    --model_def ../../lm_head.onnx \
    --mlir lm_head.mlir

model_deploy.py \
    --mlir lm_head.mlir \
    $quantize_args \
    --quant_input \
    --quant_output \
    --chip bm1684x \
    $device_args \
    --model lm_head.bmodel

rm *.npz

models=${models}${outdir}'/lm_head.bmodel '
popd

echo $models

outdir=$folder/$mode"_"$num_device"dev"/block
mkdir -p $outdir

pushd $outdir
mkdir -p $outdir

for ((i=0; i<=$num_layers; i++)); do

    model_transform.py \
        --model_name block_$i \
        --model_def ../../block_$i.onnx \
        --mlir block_$i.mlir

    model_deploy.py \
        --mlir block_$i.mlir \
        $quantize_args \
        --quant_input \
        --quant_output \
        --chip bm1684x \
        $device_args \
        --model block_$i.bmodel

    model_transform.py \
        --model_name block_cache_$i \
        --model_def ../../block_cache_$i.onnx \
        --mlir block_cache_$i.mlir

    model_deploy.py \
        --mlir block_cache_$i.mlir \
        $quantize_args \
        --quant_input \
        --quant_output \
        --chip bm1684x \
        $device_args \
        $addr_mode_flag \
        --model block_cache_$i.bmodel

    rm *.npz

    models=${models}${outdir}'/block_'$i'.bmodel '$outdir'/block_cache_'$i'.bmodel '

done
popd
echo $models

outdir=./models/BM1684X/
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi

model_tool --combine $models -o ${outdir}${out_model}
