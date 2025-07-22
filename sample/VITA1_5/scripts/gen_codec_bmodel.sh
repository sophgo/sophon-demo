#!/bin/bash
scripts_dir=$(dirname $(readlink -f "$0"))
pushd $scripts_dir

set -ex
models=""
mode=int4
mode_args=""
device_args=""
quantize_args=""
name="codec"
addr_mode="io_alone"
chip="bm1684x"
num_core="1"
num_layers=
num_pre_nn_layers=2
hidden_size=
out_model=$name.bmodel
onnx_dir=$scripts_dir/../tools/tmp/onnx_tts/

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

if [ "$name" = "codec" ]; then
  num_layers=4
  num_pre_nn_layers=2
  hidden_size=896
  echo "Compile vita-codec"
else
  >&2 echo -e "Error: Invalid name $name"
  exit 1
fi

if [ x$mode == x"int8" ]; then
    quantize_args="--quantize W8F16 --q_group_size 64"
elif [ x$mode == x"bf16" ]; then
    quantize_args="--quantize F16"
elif [ x$mode == x"fp16" ]; then
    quantize_args="--quantize F16"
elif [ x$mode == x"fp32" ]; then
    quantize_args="--quantize F32"
elif [ x$mode == x"int4" ]; then
    quantize_args="--quantize W4F16 --q_group_size 64"
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

# convert embedding
outdir=tmp/embedding
mkdir -p $outdir
pushd $outdir

model_transform.py \
    --model_name tts_embedding \
    --model_def ${onnx_dir}/tts_embedding.onnx \
    --input_shapes [[1,1]] \
    --input_types "int32" \
    --mlir tts_embedding.mlir

model_deploy.py \
    --mlir tts_embedding.mlir \
    --quantize F16 \
    --quant_input \
    --quant_output \
    --chip ${chip} \
    $device_args \
    --num_core $num_core \
    --model tts_embedding.bmodel

rm *.npz -f

models=$models' '$outdir'/tts_embedding.bmodel '

popd

echo $models

outdir=tmp/lm_head
mkdir -p $outdir
pushd $outdir

model_transform.py \
    --model_name lm_head \
    --input_shapes [[1,1,$hidden_size]] \
    --model_def ${onnx_dir}/lm_head.pt \
    --mlir lm_head.mlir

model_deploy.py \
    --mlir lm_head.mlir \
    --quantize F16 \
    --quant_input \
    --chip ${chip} \
    $device_args \
    --num_core $num_core \
    --model lm_head.bmodel

rm *.npz -f

models=${models}${outdir}'/lm_head.bmodel '
popd

echo $models

outdir=${folder}/pre_nn_layers
mkdir -p $outdir
pushd $outdir

for ((i=0; i<$num_pre_nn_layers; i++)); do
    model_transform.py \
        --model_name pre_$i \
        --model_def ${onnx_dir}/layer_pre_nn_$i.onnx \
        --mlir layer_pre_nn_$i.mlir

    model_deploy.py \
        --mlir layer_pre_nn_$i.mlir \
        $quantize_args \
        --quant_input \
        --quant_output \
        --chip ${chip} \
        $device_args \
        --num_core $num_core \
        --model layer_pre_nn_$i.bmodel
    rm *.npz -f
    models=${models}${outdir}'/layer_pre_nn_'$i'.bmodel '
done

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