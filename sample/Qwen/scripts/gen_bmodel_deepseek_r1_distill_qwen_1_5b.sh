#!/bin/bash
set -ex
models=
mode="f16"
folder="models"
num_device=1
device_args=""
addr_args=""
dyn_args=""
quantize_args="--quantize F16"
name=""
num_layers=
hidden_size=
seq_length=
out_model=$name.bmodel
dynamic=0

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

if [ "$name" = "qwen2.5-1.5b" ]; then
  num_layers=28
  hidden_size=1536
  echo "Compile Qwen2.5-1.5B"
else
  >&2 echo -e "Error: Invalid name $name, the input name must be \033[31mqwen2.5-1.5b\033[0m"
  exit 1
fi

if [[ -z "$seq_length" ]]; then
    echo "Error: --seq_length is required." >&2
    exit 1
fi

if [ x$mode == x"int8" ]; then
    quantize_args="--quantize W8F16"
elif [ x$mode == x"f16" ]; then
    quantize_args="--quantize F16"
elif [ x$mode == x"int4" ]; then
    quantize_args="--quantize W4F16 --q_group_size 64"
else
    echo "Error, unknown quantize mode"
    exit 1
fi

if [ x$addr_mode == x"io_alone" ]; then
    addr_args="--addr_mode io_alone"
fi

out_model=$name'_'$mode'_seq'$seq_length'_1688_2core.bmodel'


outdir=${folder}/embedding
mkdir -p $outdir
pushd $outdir

model_transform.py \
    --model_name embedding \
    --model_def ../onnx/embedding.pt \
    --input_shapes "[[1,$seq_length]]" \
    --input_types "int32" \
    --mlir embedding.mlir

model_deploy.py \
    --mlir embedding.mlir \
    --quantize F16 \
    --quant_output \
    --chip bm1688 \
    --num_core 2 \
    $device_args \
    $dyn_args \
    --model embedding.bmodel

model_transform.py \
    --model_name embedding_cache \
    --model_def ../onnx/embedding.pt \
    --input_shapes "[[1,1]]" \
    --input_types "int32" \
    --mlir embedding_cache.mlir

model_deploy.py \
    --mlir embedding_cache.mlir \
    --quantize F16 \
    --num_core 2 \
    --quant_output \
    --chip bm1688 \
    $device_args \
    --model embedding_cache.bmodel

rm *.npz

models=$models' '$outdir'/embedding.bmodel '$outdir'/embedding_cache.bmodel '

popd

echo $models

outdir=${folder}/$mode/lm_head
mkdir -p $outdir
pushd $outdir

model_transform.py \
    --model_name lm_head \
    --model_def ../../onnx/lm_head_with_topk.pt \
    --input_shapes "[[1,1,${hidden_size}]]" \
    --mlir lm_head.mlir

model_deploy.py \
    --mlir lm_head.mlir \
    ${quantize_args} \
    --quant_input \
    --chip bm1688 \
    --num_core 2 \
    --model lm_head_with_topk.bmodel

models=${models}${outdir}'/lm_head_with_topk.bmodel '


popd
echo $models

outdir=${folder}/$mode/block
mkdir -p $outdir
pushd $outdir

# Function to process each block in parallel
process_block() {
    i=$1

    model_transform.py \
        --model_name block_$i \
        --model_def ../../onnx/block_$i.onnx \
        --mlir block_$i.mlir

    model_deploy.py \
        --mlir block_$i.mlir \
        $quantize_args \
        --quant_input \
        --quant_output \
        --chip bm1688 \
        --num_core 2 \
        $dyn_args \
        --model block_$i.bmodel

    model_transform.py \
        --model_name block_cache_$i \
        --model_def ../../onnx/block_cache_${i}.onnx \
        --mlir block_cache_$i.mlir

    model_deploy.py \
        --mlir block_cache_$i.mlir \
        $quantize_args \
        --quant_input \
        --quant_output \
        --chip bm1688 \
        --num_core 2 \
        $addr_args \
        --model block_cache_$i.bmodel
}
# Process each block in parallel
for ((i=0; i<$num_layers; i++)); do
    process_block $i &
    models=${models}${outdir}'/block_'$i'.bmodel '$outdir'/block_cache_'$i'.bmodel '
    sleep 45
done

wait  # Wait for all background processes to finish

rm -f *.npz
popd
echo $models

outdir=./models/BM1688/
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi

model_tool --combine $models -o ${outdir}${out_model}
