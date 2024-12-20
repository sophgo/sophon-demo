#!/bin/bash
set -ex

exe_dir=$(dirname $(readlink -f "$0"))
pushd $exe_dir

combined_outdir=../models/BM1684X/

if [ ! -d $combined_outdir ]; 
then
  mkdir -p $combined_outdir
else
  echo dir $combined_outdir exist
fi

models=
folder=${exe_dir}"/../models/onnx"
device_args=""
quantize_args="--quantize W4F16"
addr_args="--addr_mode io_alone"
name="qwen2-vl-7b"
num_layers=
out_model=$name.bmodel
seq_length=512
hidden_size=
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

if [ "$name" = "qwen2-vl-7b" ]; then
  num_layers=28
  hidden_size=3584
  echo "Compile Qwen2-VL-7B"
elif [ "$name" = "qwen2-vl-2b" ]; then
  num_layers=28
  hidden_size=1536
  echo "Compile Qwen2-VL-2B"
else
  >&2 echo -e "Error: Invalid name $name, the input name must be \033[31mqwen2-vl-2b|qwen2-vl-7b\033[0m"
  exit 1
fi

if [ x$mode == x"int8" ]; then
    quantize_args="--quantize W8BF16"
elif [ x$mode == x"bf16" ]; then
    quantize_args="--quantize BF16"
elif [ x$mode == x"fp16" ]; then
    quantize_args="--quantize F16"
elif [ x$mode == x"int4" ]; then
    quantize_args="--quantize W4BF16 --q_group_size 32"
else
    echo "Error, unknown quantize mode"
    exit 1
fi

timestamp=$(date "+%Y%m%d_%H%M%S")
out_model=${name}_${mode}_seq${seq_length}_1dev_${timestamp}.bmodel


if [ x$addr_mode == x"io_alone" ]; then
    addr_args="--addr_mode io_alone"
fi

outdir=${folder}/$mode"_1dev"/embedding
mkdir -p $outdir
pushd $outdir

model_transform.py \
    --model_name embedding \
    --model_def ../../llm/embedding.pt \
    --input_shapes [[1,${seq_length}]] \
    --input_types "int32" \
    --mlir embedding.mlir

model_deploy.py \
    --mlir embedding.mlir \
    ${quantize_args} \
    --quant_input \
    --quant_output \
    --chip bm1684x \
    $device_args \
    $dyn_args \
    --model embedding.bmodel

model_transform.py \
    --model_name embedding_cache \
    --model_def ../../llm/embedding.pt \
    --input_shapes [[1,1]] \
    --input_types "int32" \
    --mlir embedding_cache.mlir

model_deploy.py \
    --mlir embedding_cache.mlir \
    ${quantize_args} \
    --quant_input \
    --quant_output \
    --chip bm1684x \
    $device_args \
    --model embedding_cache.bmodel

models=$models' '$outdir'/embedding.bmodel '$outdir'/embedding_cache.bmodel '

rm -f *.npz
popd
echo $models

outdir=${folder}/$mode"_1dev"/lm_head
mkdir -p $outdir
pushd $outdir

model_transform.py \
    --model_name lm_head \
    --model_def ../../llm/lm_head.pt \
    --input_shapes [[1,${hidden_size}]] \
    --mlir lm_head.mlir

model_deploy.py \
    --mlir lm_head.mlir \
    ${quantize_args} \
    --quant_input \
    --chip bm1684x \
    $device_args \
    --model lm_head.bmodel

model_transform.py \
    --model_name greedy_head \
    --model_def ../../llm/greedy_head.onnx \
    --mlir greedy_head.mlir

model_deploy.py \
    --mlir greedy_head.mlir \
    --chip bm1684x \
    --model greedy_head.bmodel


model_transform.py \
    --model_name penalty_sample_head \
    --model_def ../../llm/penalty_sample_head.onnx \
    --mlir penalty_sample_head.mlir

model_deploy.py \
    --mlir penalty_sample_head.mlir \
    --chip bm1684x \
    --model penalty_sample_head.bmodel
    
    
models=${models}${outdir}'/lm_head.bmodel '$outdir'/greedy_head.bmodel '$outdir'/penalty_sample_head.bmodel '

rm -f *.npz
popd
echo $models

outdir=${folder}/$mode"_1dev"/block
mkdir -p $outdir
echo $outdir
pushd $outdir

for ((i=0; i<$num_layers; i++)); do

    model_transform.py \
        --model_name block_$i \
        --model_def ../../llm/block_$i.onnx \
        --mlir block_$i.mlir

    model_deploy.py \
        --mlir block_$i.mlir \
        ${quantize_args} \
        --quant_input \
        --quant_output \
        --chip bm1684x \
        $device_args \
        $dyn_args \
        --model block_$i.bmodel

    model_transform.py \
        --model_name block_cache_$i \
        --model_def ../../llm/block_cache_$i.onnx \
        --mlir block_cache_$i.mlir

    model_deploy.py \
        --mlir block_cache_$i.mlir \
        ${quantize_args} \
        --quant_input \
        --quant_output \
        --chip bm1684x \
        $device_args \
        $addr_args \
        $dyn_args \
        --model block_cache_$i.bmodel

    rm -f *.npz

    models=${models}${outdir}'/block_'$i'.bmodel '$outdir'/block_cache_'$i'.bmodel '

done
popd
echo $models

# Compile VIT model
outdir=${folder}/$mode"_1dev"/vit
mkdir -p $outdir
pushd $outdir
model_transform.py \
  --model_name vit \
  --model_def ../../vit/vision_transformer.onnx \
  --input_shapes [[${vision_seq_length},1176],[${vision_seq_length},2],[1,${vision_seq_length},${vision_seq_length}]] \
  --input_types "float32,int32,float32" \
  --mlir vit.mlir 

model_deploy.py \
  --mlir vit.mlir \
  --quantize F16 \
  --chip bm1684x \
  --model vit.bmodel

popd

model_tool --combine $models $outdir/vit.bmodel -o $out_model
chmod 666 $out_model
mv $out_model $combined_outdir