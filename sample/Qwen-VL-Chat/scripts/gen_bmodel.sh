#!/bin/bash
set -ex

exe_dir=$(dirname $(readlink -f "$0"))
pushd $exe_dir

combined_outdir=../models/BM1684X/

models=
mode="int8"
folder=${exe_dir}"/../models/onnx"
num_device=1
device_args=""
quantize_args="--quantize W8BF16"
addr_args=""
dyn_args=""
name="qwen-vl-chat"
num_layers=
out_model=$name.bmodel
seq_length=
hidden_size=

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

if [ "$name" = "qwen-vl-chat" ]; then
  num_layers=32
  hidden_size=4096
  echo "Compile Qwen-VL-Chat"
else
  >&2 echo -e "Error: Invalid name $name"
  exit 1
fi

if [ x$mode == x"int8" ]; then
    quantize_args="--quantize W8F16"
elif [ x$mode == x"bf16" ]; then
    quantize_args="--quantize BF16"
elif [ x$mode == x"int4" ]; then
    quantize_args="--quantize W4F16 --q_group_size 64"
else
    echo "Error, unknown quantize mode"
    exit 1
fi

if [ x$num_device != x1 ]; then
    device_args="--num_device $num_device"
    out_model=${name}-${mode}-vit-fp16-${num_device}dev.bmodel
else
    out_model=${name}-${mode}-vit-fp16-1dev.bmodel
fi

if [ x$addr_mode == x"io_alone" ]; then
    addr_args="--addr_mode io_alone"
fi

outdir=${folder}/$mode"_"$num_device"dev"/embedding
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
    --quantize F16 \
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
    --quantize F16 \
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

if [[ $num_device -eq 1 ]]; then

    model_transform.py \
        --model_name lm_head \
        --model_def ../../llm/lm_head.pt \
        --input_shapes [[1,${hidden_size}]] \
        --mlir lm_head.mlir
    
    model_deploy.py \
        --mlir lm_head.mlir \
        $quantize_args \
        --quant_input \
        --chip bm1684x \
        $device_args \
        --model lm_head.bmodel

    models=${models}${outdir}'/lm_head.bmodel '
fi

rm -f *.npz
popd
echo $models


outdir=${folder}/$mode"_"$num_device"dev"/block
mkdir -p $outdir
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

    rm -f *.npz

    models=${models}${outdir}'/block_'$i'.bmodel '

done
popd
echo $models


outdir=${folder}/$mode"_"$num_device"dev"/block
mkdir -p $outdir
pushd $outdir

for ((i=0; i<$num_layers; i++)); do

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

    models=${models}$outdir'/block_cache_'$i'.bmodel '

done
popd
echo $models

outdir=${folder}/$mode"_"$num_device"dev"/lm_head
mkdir -p $outdir
pushd $outdir

if [[ $num_device -gt 1 ]]; then
    model_transform.py \
        --model_name lm_head \
        --model_def ../../llm/lm_head_with_topk.pt \
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
    
    models=${models}${outdir}'/penalty_sample_head.bmodel '$outdir'/greedy_head.bmodel ' 
fi

rm -f *.npz
popd
echo $models

outdir=${folder}/$mode"_"$num_device"dev"/vit
mkdir -p $outdir
pushd $outdir

model_transform.py \
    --model_name qwen_vit \
    --model_def ../../vit/vision_transformer.onnx \
    --input_shapes [[1,3,448,448]] \
    --input_types "int32" \
    --mlir qwen_vit.mlir

model_deploy.py \
    --mlir qwen_vit.mlir \
    --quantize F16 \
    --quant_output \
    --chip bm1684x \
    --model qwen_vit_1684x_f16.bmodel \
    --compare_all

models=${outdir}'/qwen_vit_1684x_f16.bmodel '${models}

rm -f *.npz
popd
echo $models

model_tool --combine $models -o $out_model
chmod a+r $out_model

if [ ! -d $combined_outdir ]; then
    mkdir -p $combined_outdir
fi

mv $out_model $combined_outdir

popd