 #!/bin/bash 
#   ./compile.sh --mode int8 --name chattts-llama --seq_length 512 # same as int4
script_dir=$(dirname $(readlink -f "$0"))
pushd $script_dir

set -ex
models=
mode="bf16"
folder="tmp"
num_device=1
mode_args=""
device_args=""
addr_args=""
quantize_args="--quantize F16"
name="chattts-llama"
num_layers=
hidden_size=
seq_length=
out_model=$name.bmodel
target=bm1684x
onnx_path=$script_dir/../models/onnx/gpt

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

if [ "$name" = "chattts-llama" ]; then
  num_layers=20
  hidden_size=768
  echo "Compile chattts-llama"
else
  >&2 echo -e "Error: Invalid name $name, the input name must be \033[31mllama3-8b\033[0m"
  exit 1
fi

if [ x$mode == x"int8" ]; then
    quantize_args="--quantize W8F16"
elif [ x$mode == x"bf16" ]; then
    quantize_args="--quantize F16"
elif [ x$mode == x"int4" ]; then
    quantize_args="--quantize W4F16 --q_group_size 64"
else
    echo "Error, unknown quantize mode"
    exit 1
fi

if [ x$num_device != x1 ]; then
    device_args="--num_device $num_device"
    out_model=$name'_'$mode'_'$num_device'dev_'$seq_length'_'$target'.bmodel'
else
    out_model=$name'_'$mode'_1dev_'$seq_length'_'$target'.bmodel'
fi

if [ x$addr_mode == x"io_alone" ]; then
    addr_args="--addr_mode io_alone"
fi

outdir=${folder}/embedding
mkdir -p $outdir
pushd $outdir

model_transform.py \
    --model_name embedding_text \
    --model_def $onnx_path/embedding_text.onnx \
    --mlir embedding_text.mlir

model_deploy.py \
    --mlir embedding_text.mlir \
    --quantize F16 \
    --quant_input \
    --quant_output \
    --chip $target \
    $device_args \
    --model embedding_text.bmodel

model_transform.py \
    --model_name embedding_text_cache \
    --model_def $onnx_path/embedding_text.onnx \
    --input_shapes [[1,1]] \
    --mlir embedding_text_cache.mlir

model_deploy.py \
    --mlir embedding_text_cache.mlir \
    --quantize F16 \
    --quant_input \
    --quant_output \
    --chip $target \
    $device_args \
    --model embedding_text_cache.bmodel

# model_transform.py \
#     --model_name embedding_code \
#     --model_def $onnx_path/embedding_code.onnx \
#     --mlir embedding_code.mlir

# model_deploy.py \
#     --mlir embedding_code.mlir \
#     --quantize F16 \
#     --quant_input \
#     --quant_output \
#     --chip $target \
#     $device_args \
#     --model embedding_code.bmodel

model_transform.py \
    --model_name embedding_code_cache \
    --model_def $onnx_path/embedding_code_cache.onnx \
    --mlir embedding_code_cache.mlir

model_deploy.py \
    --mlir embedding_code_cache.mlir \
    --quantize F16 \
    --quant_input \
    --quant_output \
    --chip $target \
    $device_args \
    --model embedding_code_cache.bmodel
rm *.npz

models=$models' '$outdir'/embedding_text.bmodel '$outdir'/embedding_text_cache.bmodel '$outdir'/embedding_code.bmodel '$outdir'/embedding_code_cache.bmodel '

popd

echo $models

outdir=${folder}/$mode"_"$num_device"dev"/lm_head
mkdir -p $outdir
pushd $outdir

model_transform.py \
    --model_name lm_head_text \
    --model_def $onnx_path/lm_head_text.onnx \
    --input_shapes [[1,${hidden_size}]] \
    --mlir lm_head_text.mlir

model_deploy.py \
    --mlir lm_head_text.mlir \
    $quantize_args \
    --quant_input \
    --chip $target \
    $device_args \
    --model lm_head_text.bmodel

model_transform.py \
    --model_name lm_head_code \
    --model_def $onnx_path/lm_head_code.onnx \
    --input_shapes [[1,${hidden_size}]] \
    --mlir lm_head_code.mlir

model_deploy.py \
    --mlir lm_head_code.mlir \
    $quantize_args \
    --quant_input \
    --chip $target \
    $device_args \
    --model lm_head_code.bmodel

# model_transform.py \
#     --model_name penalty_sample_head_text \
#     --model_def ../../onnx/penalty_sample_head_text.onnx \
#     --mlir penalty_sample_head_text.mlir

# model_deploy.py \
#     --mlir penalty_sample_head_text.mlir \
#     --chip $target \
#     --model penalty_sample_head_text.bmodel

# model_transform.py \
#     --model_name chattts_sample_head_code \
#     --model_def ../../onnx/chattts_sample_head_code.onnx \
#     --mlir chattts_sample_head_code.mlir \
#     --test_input ./../onnx/chattts_sample_head_code_input.npz

# model_deploy.py \
#     --mlir chattts_sample_head_code.mlir \
#     --chip $target \
#     --model chattts_sample_head_code.bmodel

model_transform.py \
    --model_name greedy_head_text \
    --model_def $onnx_path/greedy_head_text.onnx \
    --mlir greedy_head_text.mlir

model_deploy.py \
    --mlir greedy_head_text.mlir \
    --chip $target \
    --model greedy_head_text.bmodel

model_transform.py \
    --model_name greedy_head_code \
    --model_def $onnx_path/greedy_head_code.onnx \
    --mlir greedy_head_code.mlir

model_deploy.py \
    --mlir greedy_head_code.mlir \
    --chip $target \
    --model greedy_head_code.bmodel
rm *.npz

# models=${models}${outdir}'/lm_head_text.bmodel '$outdir'/lm_head_code.bmodel '$outdir'/penalty_sample_head_text.bmodel '$outdir'/chattts_sample_head_code.bmodel '
models=${models}${outdir}'/lm_head_text.bmodel '$outdir'/lm_head_code.bmodel '$outdir'/greedy_head_text.bmodel '$outdir'/greedy_head_code.bmodel '
popd

echo $models

outdir=${folder}/$mode"_"$num_device"dev"/block
mkdir -p $outdir

pushd $outdir
mkdir -p $outdir

for ((i=0; i<$num_layers; i++)); do
    model_transform.py \
        --model_name block_$i \
        --model_def $onnx_path/block_$i.onnx \
        --mlir block_$i.mlir

    model_deploy.py \
        --mlir block_$i.mlir \
        $quantize_args \
        --quant_input \
        --quant_output \
        --chip $target \
        $device_args \
        --model block_$i.bmodel

    model_transform.py \
        --model_name block_cache_$i \
        --model_def $onnx_path/block_cache_$i.onnx \
        --mlir block_cache_$i.mlir

    model_deploy.py \
        --mlir block_cache_$i.mlir \
        $quantize_args \
        --quant_input \
        --quant_output \
        --chip $target \
        $device_args \
        $addr_args \
        --model block_cache_$i.bmodel

    rm *.npz

    models=${models}${outdir}'/block_'$i'.bmodel '$outdir'/block_cache_'$i'.bmodel '

done
popd
echo $models

model_tool --combine $models -o $out_model

popd