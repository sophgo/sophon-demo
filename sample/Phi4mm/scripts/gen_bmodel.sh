#!/bin/bash
scripts_dir=$(dirname $(readlink -f "$0"))
pushd $scripts_dir

set -ex
models=""
mode=int4
mode_args=""
device_args=""
quantize_args=""
name="phi4"
addr_mode="io_alone"
chip="bm1684x"
num_core="1"
num_layers=
hidden_size=
out_model=$name.bmodel
onnx_dir=$scripts_dir/../tools/tmp/onnx_seq512/

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
    --name)
        name="$2"
        shift 2
        ;;
    --num_core)
        num_core="$2"
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

if [ "$name" = "phi4mm" ]; then
  num_layers=32
  hidden_size=3072
  echo "Compile phi4mm"
else
  >&2 echo -e "Error: Invalid name $name"
  exit 1
fi

if [ x$mode == x"int8" ]; then
    quantize_args="--quantize W8BF16 --q_group_size 64"
elif [ x$mode == x"bf16" ]; then
    quantize_args="--quantize BF16"
elif [ x$mode == x"fp16" ]; then
    quantize_args="--quantize F16"
elif [ x$mode == x"int4" ]; then
    quantize_args="--quantize W4BF16 --q_group_size 64"
else
    echo "Error, unknown quantize mode"
    exit 1
fi

folder='tmp/'$name'_'$chip'_'$mode'_'$num_core'core'
out_model=$name'_'$chip'_'$mode'_'$num_core'core.bmodel'

# Compile greedy head model
outdir=tmp/greedy
mkdir -p $outdir
pushd $outdir
model_transform.py \
    --model_name greedy_head \
    --model_def ${onnx_dir}/greedy_head.onnx \
    --mlir greedy_head.mlir

model_deploy.py \
    --mlir greedy_head.mlir \
    --chip ${chip} \
    --model greedy_head.bmodel

models=${models}${outdir}'/greedy_head.bmodel '
popd

echo $models

# Compile VIT model
outdir=tmp/vit
mkdir -p $outdir
pushd $outdir

model_transform.py \
    --model_name phi4mm_vit \
    --model_def ${onnx_dir}/vit.onnx \
    --input_shapes [[1,3,448,448]] \
    --mlir phi4mm_vit.mlir \

model_deploy.py \
    --mlir phi4mm_vit.mlir \
    --quantize BF16 \
    --processor ${chip} \
    --quant_output \
    --model vit_bf16.bmodel \

models=${models}${outdir}'/vit_bf16.bmodel '

popd
echo $models

# Compile speech model
outdir=tmp/speech
mkdir -p $outdir
pushd $outdir

model_transform.py \
    --model_name phi4mm_speech \
    --model_def ${onnx_dir}/speech.onnx \
    --input_shapes [[1,384,80]] \
    --mlir phi4mm_speech.mlir

model_deploy.py \
    --mlir phi4mm_speech.mlir \
    --quantize BF16 \
    --processor ${chip} \
    --quant_output \
    --model speech_bf16.bmodel

models=${models}${outdir}'/speech_bf16.bmodel '

popd
echo $models

# convert embedding
outdir=tmp/embedding
mkdir -p $outdir
pushd $outdir

model_transform.py \
    --model_name embedding \
    --input_shapes [[1,512]] \
    --model_def ${onnx_dir}/embedding.pt \
    --input_types "int32" \
    --mlir embedding.mlir

model_deploy.py \
    --mlir embedding.mlir \
    --quantize BF16 \
    --quant_input \
    --quant_output \
    --chip ${chip} \
    $device_args \
    --num_core $num_core \
    --model embedding.bmodel

model_transform.py \
    --model_name embedding_cache \
    --model_def ${onnx_dir}/embedding.pt \
    --input_shapes [[1,1]] \
    --input_types "int32" \
    --mlir embedding_cache.mlir

model_deploy.py \
    --mlir embedding_cache.mlir \
    --quantize BF16 \
    --quant_input \
    --quant_output \
    --chip ${chip} \
    $device_args \
    --num_core $num_core \
    --model embedding_cache.bmodel

rm *.npz -f

models=$models' '$outdir'/embedding.bmodel '$outdir'/embedding_cache.bmodel '

popd

echo $models

outdir=tmp/lm_head
mkdir -p $outdir
pushd $outdir

model_transform.py \
    --model_name lm_head \
    --input_shapes [[1,$hidden_size]] \
    --model_def ${onnx_dir}/lm_head.pt \
    --mlir lm_head.mlir

model_deploy.py \
    --mlir lm_head.mlir \
    --quantize BF16 \
    --quant_input \
    --chip ${chip} \
    $device_args \
    --num_core $num_core \
    --model lm_head.bmodel

rm *.npz -f

models=${models}${outdir}'/lm_head.bmodel '
popd

echo $models

outdir=${folder}/block
mkdir -p $outdir
pushd $outdir


for ((i=0; i<$num_layers; i++)); do
    model_transform.py \
        --model_name block_$i \
        --model_def ${onnx_dir}/block_$i.onnx \
        --mlir block_$i.mlir

    model_deploy.py \
        --mlir block_$i.mlir \
        $quantize_args \
        --quant_input \
        --quant_output \
        --chip ${chip} \
        $device_args \
        --num_core $num_core \
        --model block_$i.bmodel
        # --quantize_table ${onnx_dir}/decoder_qtable \

    model_transform.py \
        --model_name block_cache_$i \
        --model_def ${onnx_dir}/block_cache_$i.onnx \
        --mlir block_cache_$i.mlir

    model_deploy.py \
        --mlir block_cache_$i.mlir \
        $quantize_args \
        --quant_input \
        --quant_output \
        --chip ${chip} \
        $device_args \
        --num_core $num_core \
        --addr_mode io_alone \
        --model block_cache_$i.bmodel
        # --quantize_table ${onnx_dir}/decoder_qtable \

    rm *.npz -f

    models=${models}${outdir}'/block_'$i'.bmodel '$outdir'/block_cache_'$i'.bmodel '

done
popd
echo $models

model_tool --combine $models -o $out_model

popd