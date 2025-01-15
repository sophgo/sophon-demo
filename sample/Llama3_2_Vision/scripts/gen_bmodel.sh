#!/bin/bash
set -ex

scripts_dir=$(dirname $(readlink -f "$0"))
pushd $scripts_dir

combined_outdir=../models/BM1684X/

if [ ! -d $combined_outdir ]; 
then
  mkdir -p $combined_outdir
else
  echo dir $combined_outdir exist
fi

models=
folder=${scripts_dir}"/../models/onnx"
device_args=""
quantize_args="--quantize W4F16"
addr_args="--addr_mode io_alone"
name="llama3.2-11b"
num_layers=40
out_model=$name.bmodel
seq_length=512
hidden_size=4096
mode="int4"
num_device=1
vision_seq_length=1024

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
    --mode)
        mode="$2"
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
    --vision_seq_length)
        vision_seq_length="$2"
        shift 2
        ;;
    --dynamic)
        dynamic="$2"
        shift 2
        ;;
    --target)
        target="$2"
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

if [ "$name" = "llama3.2-11b" ]; then
  num_layers=40
  hidden_size=4096
  echo "Compile Llama3.2-11B-Vision"
else
  echo -e "Error: Invalid name $name, the input name must be \033[31mllama3.2-11b\033[0m"
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

timestamp=$(date "+%Y%m%d_%H%M%S")

if [ x$num_device != x1 ]; then
    device_args="--num_device $num_device"
    out_model=$name'_'$mode'_'$num_device'dev_'$seq_length'seq.bmodel'
else
    out_model=$name'-vision_'$mode'_'$seq_length'seq.bmodel'
fi

if [ x$addr_mode == x"io_alone" ]; then
    addr_args="--addr_mode io_alone"
fi

outdir=${folder}/embedding
mkdir -p $outdir
pushd $outdir

model_transform.py \
    --model_name embedding \
    --model_def ../embedding.pt \
    --input_shapes "[[1,$seq_length]]" \
    --input_types "int32" \
    --mlir embedding.mlir

model_deploy.py \
    --mlir embedding.mlir \
    --quantize BF16 \
    --quant_output \
    --chip bm1684x \
    $device_args \
    --model embedding.bmodel

model_transform.py \
    --model_name embedding_cache \
    --model_def ../embedding.pt \
    --input_shapes "[[1,1]]" \
    --input_types "int32" \
    --mlir embedding_cache.mlir

model_deploy.py \
    --mlir embedding_cache.mlir \
    --quantize BF16 \
    --quant_output \
    --chip bm1684x \
    $device_args \
    --model embedding_cache.bmodel

rm *.npz -f
models=$models' '$outdir'/embedding.bmodel '$outdir'/embedding_cache.bmodel '
popd


echo $models

outdir=${folder}/$mode"_"$num_device"dev"/lm_head
mkdir -p $outdir
pushd $outdir

model_transform.py \
    --model_name lm_head \
    --model_def ../../lm_head.pt \
    --input_shapes "[[1,${hidden_size}]]" \
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
    --model_def ../../greedy_head.onnx \
    --mlir greedy_head.mlir

model_deploy.py \
    --mlir greedy_head.mlir \
    --chip bm1684x \
    --model greedy_head.bmodel


model_transform.py \
    --model_name penalty_sample_head \
    --model_def ../../penalty_sample_head.onnx \
    --mlir penalty_sample_head.mlir

model_deploy.py \
    --mlir penalty_sample_head.mlir \
    --chip bm1684x \
    --model penalty_sample_head.bmodel

rm *.npz -f
models=${models}${outdir}'/lm_head.bmodel '$outdir'/greedy_head.bmodel '$outdir'/penalty_sample_head.bmodel '
popd

echo $models

outdir=${folder}/$mode"_"$num_device"dev"/block
mkdir -p $outdir
pushd $outdir
process_block()
{
    i=$1
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
        $addr_args \
        --model block_cache_$i.bmodel
    
}

# Process each block in parallel
for ((i=0; i<$num_layers; i++)); do
    process_block $i &
    sleep 45
done
wait  # Wait for all background processes to finish
rm *.npz -f
popd


echo $models

# combine order should separate cross block to avoid device addr assign error
cross_attn_layer=(3 8 13 18 23 28 33 38)
for ((i=0; i<$num_layers; i++)); do
    layers_str=" ${cross_attn_layer[*]} "
    if [[ $layers_str =~ " $i " ]]; then
        echo $i is cross block
    else
        models=${models}${outdir}'/block_'$i'.bmodel '$outdir'/block_cache_'$i'.bmodel '
    fi
done

for i in "${cross_attn_layer[@]}"; do
    models=${models}${outdir}'/block_'$i'.bmodel '$outdir'/block_cache_'$i'.bmodel '
done

echo $models

outdir=${folder}/vit
mkdir -p $outdir
pushd $outdir

model_transform.py \
    --model_name vit \
    --model_def ../vit/vision_transformer.onnx \
    --mlir vit.mlir

model_deploy.py \
    --mlir vit.mlir \
    --quantize BF16 \
    --quant_output \
    --chip bm1684x \
    --model vit.bmodel

# rm *.npz *.onnx -f
models=${models}${outdir}'/vit.bmodel '
popd

echo $models

model_tool --combine $models -o $out_model

chmod 666 $out_model
mv $out_model $combined_outdir
popd