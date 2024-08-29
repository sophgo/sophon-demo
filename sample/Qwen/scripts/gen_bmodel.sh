#!/bin/bash
set -ex
models=
mode="int8"
folder="./models"
num_device=1
mode_args=""
device_args=""
quantize_args="--quantize W8BF16"
addr_args=""
dyn_args=""
name=""
num_layers=
out_model=$name.bmodel
seq_length=
hidden_size=
dynamic=0
batch=1

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
    --addr_mode)
        addr_mode="$2"
        shift 2
        ;;
    --seq_length)
        seq_length="$2"
        shift 2
        ;;
    --dynamic)
        dynamic="$2"
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

if [[ -z "$seq_length" ]]; then
    echo "Error: --seq_length is required." >&2
    exit 1
fi


if [ "$name" = "qwen1.5-7b" ]; then
  num_layers=32
  hidden_size=4096
  echo "Compile Qwen1.5-7B"
elif [ "$name" = "qwen1.5-4b" ]; then
  num_layers=40
  hidden_size=2560
  echo "Compile Qwen1.5-4B"
elif [ "$name" = "qwen1.5-1.8b" ]; then 
  num_layers=24
  hidden_size=2048
  echo "Compile Qwen1.5-1.8B"
elif [ "$name" = "qwen1.5-0.5b" ]; then 
  num_layers=24
  hidden_size=1024
  echo "Compile Qwen1.5-0.5B"
elif [ "$name" = "qwen1.5-32b" ]; then
  num_layers=64
  hidden_size=5120
  echo "Compile Qwen1.5-32B"
elif [ "$name" = "qwen1.5-14b" ]; then
  num_layers=40
  hidden_size=5120
  echo "Compile Qwen1.5-32B"
elif [ "$name" = "qwen-7b" ]; then
  num_layers=32
  hidden_size=4096
  echo "Compile Qwen-7B"
elif [ "$name" = "qwen-14b" ]; then
  num_layers=40
  hidden_size=5120
  echo "Compile Qwen-14B"
elif [ "$name" = "qwen-72b" ]; then
  num_layers=80
  hidden_size=8192
  echo "Compile Qwen-72B"
elif [ "$name" = "qwen2-7b" ]; then
  num_layers=28
  hidden_size=3584
  qwen2_arg="--disable_layer_group" 
else
  >&2 echo -e "Error: Invalid name $name, the input name must be \033[31mqwen1.5-0.5b|qwen1.5-1.8b|qwen1.5-4b|qwen1.5-7b|qwen1.5-32b\033[0m"
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
    out_model=${name}_${mode}_seq${seq_length}_${num_device}dev.bmodel
else
    out_model=${name}_${mode}_seq${seq_length}_1dev.bmodel
fi

if [ x$addr_mode == x"io_alone" ]; then
    addr_args="--addr_mode io_alone"
fi

if [ x$dynamic == x1 ]; then
    dyn_args="--dynamic"
    out_model=${name}_${mode}_seq${seq_length}_${num_device}dev_dyn.bmodel
fi

process_block() {
    i=$1
    model_transform.py \
    --model_name block_$i \
    --model_def ../../onnx/block_$i.onnx \
    --mlir block_$i.mlir

    model_deploy.py \
        --mlir block_$i.mlir \
        ${quantize_args} \
        --quant_input \
        --quant_output \
        --chip bm1684x \
        $device_args \
        $dyn_args \
        $qwen2_arg \
        --model block_$i.bmodel

    model_transform.py \
        --model_name block_cache_$i \
        --model_def ../../onnx/block_cache_$i.onnx \
        --mlir block_cache_$i.mlir

    model_deploy.py \
        --mlir block_cache_$i.mlir \
        ${quantize_args} \
        --quant_input \
        --quant_output \
        --chip bm1684x \
        $device_args \
        $dyn_args \
        $addr_args \
        --model block_cache_$i.bmodel
}

outdir=${folder}/$mode"_"$num_device"dev"/embedding
mkdir -p $outdir
pushd $outdir

model_transform.py \
    --model_name embedding \
    --model_def ../../onnx/embedding.pt \
    --input_shapes [[1,${seq_length}]] \
    --input_types "int32" \
    --mlir embedding.mlir

model_deploy.py \
    --mlir embedding.mlir \
    --quantize BF16 \
    --quant_input \
    --quant_output \
    --chip bm1684x \
    $device_args \
    $dyn_args \
    --model embedding.bmodel

model_transform.py \
    --model_name embedding_cache \
    --model_def ../../onnx/embedding.pt \
    --input_shapes [[1,1]] \
    --input_types "int32" \
    --mlir embedding_cache.mlir

model_deploy.py \
    --mlir embedding_cache.mlir \
    --quantize BF16 \
    --quant_input \
    --quant_output \
    --chip bm1684x \
    $device_args \
    --model embedding_cache.bmodel

models=$models' '$outdir'/embedding.bmodel '$outdir'/embedding_cache.bmodel '

rm -f *.npz
popd

echo $models

outdir=${folder}/$mode"_"$num_device"dev"/lm_head
mkdir -p $outdir
pushd $outdir

if [[ $num_device -gt 1 ]]; then
    model_transform.py \
        --model_name lm_head \
        --model_def ../../onnx/lm_head_with_topk.pt \
        --input_shapes [[1,1,${hidden_size}]] \
        --mlir lm_head.mlir

    model_deploy.py \
        --mlir lm_head.mlir \
        ${quantize_args} \
        --quant_input \
        --chip bm1684x \
        $device_args \
        --model lm_head.bmodel

    models=${models}${outdir}'/lm_head.bmodel '
else
    model_transform.py \
        --model_name lm_head \
        --model_def ../../onnx/lm_head.pt \
        --input_shapes [[1,${hidden_size}]] \
        --mlir lm_head.mlir
    
    model_deploy.py \
        --mlir lm_head.mlir \
        $quantize_args \
        --quant_input \
        --chip bm1684x \
        $device_args \
        --model lm_head.bmodel
    
    
    model_transform.py \
        --model_name greedy_head \
        --model_def ../../onnx/greedy_head.onnx \
        --mlir greedy_head.mlir
    
    model_deploy.py \
        --mlir greedy_head.mlir \
        --chip bm1684x \
        --model greedy_head.bmodel
    
    
    model_transform.py \
        --model_name penalty_sample_head \
        --model_def ../../onnx/penalty_sample_head.onnx \
        --mlir penalty_sample_head.mlir
    
    model_deploy.py \
        --mlir penalty_sample_head.mlir \
        --chip bm1684x \
        --model penalty_sample_head.bmodel
    
    
    models=${models}${outdir}'/lm_head.bmodel '$outdir'/greedy_head.bmodel '$outdir'/penalty_sample_head.bmodel '
fi

rm -f *.npz
popd
echo $models

outdir=${folder}/$mode"_"$num_device"dev"/block
mkdir -p $outdir
pushd $outdir
for ((i=0; i<$num_layers; i++)); do
    process_block $i
    models=${models}${outdir}'/block_'$i'.bmodel '$outdir'/block_cache_'$i'.bmodel '
done

wait
rm -f *.npz
popd
echo $models

outdir=./models/BM1684X/
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi

model_tool --combine $models -o ${outdir}${out_model}