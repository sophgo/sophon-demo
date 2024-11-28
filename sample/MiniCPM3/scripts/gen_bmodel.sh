#!/bin/bash
scripts_dir=$(dirname $(readlink -f "$0"))
pushd $scripts_dir
set -ex
models=
mode="bf16"
folder="../models"
num_device=1
device_args=""
addr_args=""
dyn_args=""
quantize_args="--quantize BF16"
name=""
num_layers=
hidden_size=
seq_length=
out_model=$name.bmodel
dynamic=0
chip_args="bm1684x"
target=bm1684x
target_dir="BM1684X"

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

target_save_path="$folder/$target_dir"
if [ ! -d "$target_save_path" ]; then
    echo "Directory $target_save_path does not exist. Creating..."
    mkdir -p "$target_save_path"
fi

if [[ -z "$seq_length" ]]; then
    echo "Error: --seq_length is required." >&2
    exit 1
fi

if [ "$name" = "minicpm3-4b" ]; then
  num_layers=62
  hidden_size=2560
  echo "Compile MiniCPM3-4B"
else
  >&2 echo -e "Error: Invalid name $name, the input name must be \033[31mminicpm3-4b\033[0m"
  exit 1
fi

if [[ -z "$seq_length" ]]; then
    echo "Error: --seq_length is required." >&2
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
    out_model=$name'_'$mode'_seq'$seq_length'_'$num_device'dev.bmodel'
else
    out_model=$name'_'$mode'_seq'$seq_length'_1dev.bmodel'
fi

if [ x$addr_mode == x"io_alone" ]; then
    addr_args="--addr_mode io_alone"
fi

if [ x$dynamic == x1 ]; then
    dyn_args="--dynamic"
    out_model=$name'_'$mode'_seq'$seq_length'_'$num_device'dev_dyn.bmodel'
fi


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
    --quantize BF16 \
    --quant_output \
    --chip $chip_args \
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
    --quantize BF16 \
    --quant_output \
    --chip $chip_args \
    $device_args \
    --model embedding_cache.bmodel

rm *.npz

models=$models' '$outdir'/embedding.bmodel '$outdir'/embedding_cache.bmodel '

popd

echo $models

outdir=${folder}/$mode"_"$num_device"dev"/lm_head
mkdir -p $outdir
pushd $outdir

if [[ $num_device -gt 1 ]]; then
    model_transform.py \
        --model_name lm_head \
        --model_def ../../onnx/lm_head_with_topk.pt \
        --input_shapes "[[1,1,${hidden_size}]]" \
        --mlir lm_head.mlir

    model_deploy.py \
        --mlir lm_head.mlir \
        ${quantize_args} \
        --quant_input \
        --chip $chip_args \
        $device_args \
        --model lm_head_with_topk.bmodel

    models=${models}${outdir}'/lm_head_with_topk.bmodel '
else
    model_transform.py \
        --model_name lm_head \
        --model_def ../../onnx/lm_head.pt \
        --input_shapes "[[1,${hidden_size}]]" \
        --mlir lm_head.mlir

    model_deploy.py \
        --mlir lm_head.mlir \
        $quantize_args \
        --quant_input \
        --chip $chip_args \
        $device_args \
        --model lm_head.bmodel

    model_transform.py \
        --model_name greedy_head \
        --model_def ../../onnx/greedy_head.onnx \
        --mlir greedy_head.mlir

    model_deploy.py \
        --mlir greedy_head.mlir \
        --chip $chip_args \
        --model greedy_head.bmodel

    model_transform.py \
        --model_name penalty_sample_head \
        --model_def ../../onnx/penalty_sample_head.onnx \
        --mlir penalty_sample_head.mlir

    model_deploy.py \
        --mlir penalty_sample_head.mlir \
        --chip $chip_args \
        --model penalty_sample_head.bmodel

    rm *.npz
    models=${models}${outdir}'/lm_head.bmodel '$outdir'/greedy_head.bmodel '$outdir'/penalty_sample_head.bmodel '
fi

popd
echo $models

outdir=${folder}/$mode"_"$num_device"dev"/block
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
        --chip $chip_args \
        $device_args \
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
        --chip $chip_args \
        $device_args \
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

model_tool --combine $models -o $out_model
if mv "$out_model" "$target_save_path/."; then
    echo "$out_model moved successfully to $target_save_path"
else
    echo "Failed to move $out_model to $target_save_path" >&2
    exit 1
fi
popd