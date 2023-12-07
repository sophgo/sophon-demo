#!/bin/bash
set -ex
models=
mode="f16"
folder="tmp"
num_device=1
mode_args=""
device_args=""
quantize_args="--quantize F16"
name=""
num_layers=
out_model=$name.bmodel

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

if [ x$mode == x"int8" ] || [ x$mode == x"int4" ]; then
    if [ x$mode == x"int8" ]; then
        quantize_args="--quantize W8F16"
    else
        quantize_args="--quantize W4F16"
    fi
    out_model=$name'_'$mode'.bmodel'
fi

if [ x$name == x"llama2-7b" ] || [ x$name == x"llama2-13b" ]; then
    if [ x$name == x"llama2-7b" ]; then
        num_layers=31
    else
        num_layers=39
    fi
fi

if [ x$num_device != x1 ]; then
    device_args="--num_device $num_device"
    out_model=$name'_'$mode'_'$num_device'dev.bmodel'
fi

outdir=${folder}/embedding
mkdir -p $outdir
pushd $outdir

seqlen=512
model_transform.py \
    --model_name embedding \
    --model_def ../embedding.onnx \
    --input_shapes [[$seqlen]] \
    --mlir embedding_${seqlen}.mlir


model_deploy.py \
    --mlir embedding_$seqlen.mlir \
    --quantize F16 \
    --chip bm1684x \
    --model embedding_${seqlen}_f16.bmodel

model_transform.py \
    --model_name embedding \
    --model_def ../embedding.onnx \
    --input_shapes [[1]] \
    --mlir embedding_1.mlir


model_deploy.py \
    --mlir embedding_1.mlir \
    --quantize F16 \
    --chip bm1684x \
    --model embedding_1_f16.bmodel

rm *.npz

models=$models' '$outdir'/embedding_1_f16.bmodel '$outdir'/embedding_'$seqlen'_f16.bmodel '

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
    --quantize F16 \
    --chip bm1684x \
    $device_args \
    --model lm_head.bmodel

rm *.npz

models=${models}${outdir}'/lm_head.bmodel '
popd

echo $models

outdir=${folder}/$mode"_"$num_device"dev"/block
mkdir -p $outdir

pushd $outdir
mkdir -p $outdir

for ((i=0; i<=$num_layers; i++))
do

model_transform.py \
    --model_name block_$i \
    --model_def ../../block_$i.onnx \
    --mlir block_$i.mlir

model_deploy.py \
    --mlir block_$i.mlir \
    $quantize_args \
    --chip bm1684x \
    --quant_output \
    --quant_output_list 2,3 \
    $device_args \
    --debug \
    --model block_$i.bmodel

model_transform.py \
    --model_name block_cache_$i \
    --model_def ../../block_cache_${i}.onnx \
    --mlir block_cache_$i.mlir

model_deploy.py \
    --mlir block_cache_$i.mlir \
    $quantize_args \
    --chip bm1684x \
    --quant_input \
    --quant_output \
    --quant_input_list 4,5 \
    --quant_output_list 2,3 \
    $device_args \
    --model block_cache_$i.bmodel

rm *.npz
# rm ../../block_$i.onnx
# rm ../../block_cache_$i.onnx

models=${models}${outdir}'/block_'$i'.bmodel '$outdir'/block_cache_'$i'.bmodel '

done
popd
echo $models

model_tool --combine $models -o $out_model