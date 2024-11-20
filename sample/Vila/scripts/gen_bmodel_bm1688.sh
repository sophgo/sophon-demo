#!/bin/bash
quantize_args="--quantize W4F16 --q_group_size 64"
addr_args="--addr_mode io_alone"
models=""
output="../BM1688/llama.bmodel"
onnx_folder="../onnx"

embedding() {
    model_transform.py \
        --model_name embedding \
        --model_def $onnx_folder/embedding.onnx \
        --mlir embedding.mlir


    model_deploy.py \
        --mlir embedding.mlir \
        --quantize F16 \
        --quant_input \
        --quant_output \
        --chip bm1688 \
        --model embedding.bmodel
}

embedding_cache() {
    model_transform.py \
        --model_name embedding_cache \
        --model_def $onnx_folder/embedding.onnx \
        --input_shape [[1,1]] \
        --mlir embedding_cache.mlir


    model_deploy.py \
        --mlir embedding_cache.mlir \
        --quantize F16 \
        --quant_input \
        --quant_output \
        --chip bm1688 \
        --model embedding_cache.bmodel
}

block() {
    i="$1"
    model_transform.py \
        --model_name block_$i \
        --model_def $onnx_folder/block_$i.onnx \
        --mlir block_$i.mlir


    model_deploy.py \
        --mlir block_$i.mlir \
        $quantize_args \
        --quant_input \
        --quant_output \
        --chip bm1688 \
        --model block_$i.bmodel
}

block_cache() {
    model_transform.py \
        --model_name block_cache_$i \
        --model_def $onnx_folder/block_cache_$i.onnx \
        --mlir block_cache_$i.mlir


    model_deploy.py \
        --mlir block_cache_$i.mlir \
        $quantize_args \
        --quant_input \
        --quant_output \
        --chip bm1688 \
        $addr_args \
        --model block_cache_$i.bmodel
}

lm_head() {
    model_transform.py \
    --model_name lm_head \
    --model_def $onnx_folder/lm_head.onnx \
    --mlir lm_head.mlir


    model_deploy.py \
        --mlir lm_head.mlir \
        $quantize_args \
        --quant_input \
        --quant_output \
        --chip bm1688 \
        --model lm_head.bmodel
}

vision_embedding() {
    model_transform.py \
        --model_name vision_embedding \
        --model_def  $onnx_folder/vision_embedding.onnx \
        --mlir vision_embedding.mlir


    model_deploy.py \
        --mlir vision_embedding.mlir \
        --quantize F16 \
        --quant_input \
        --quant_output \
        --chip bm1688 \
        --model vision_embedding.bmodel
}

mkdir -p models/BM1688
pushd "models"
mkdir tmp
pushd "tmp"
vision_embedding
mv vision_embedding.bmodel ../BM1688
embedding
embedding_cache
lm_head
models="$models embedding.bmodel embedding_cache.bmodel lm_head.bmodel"
for ((i=0; i<32; i++)); do
    block $i
    block_cache $i
    models="$models block_$i.bmodel block_cache_$i.bmodel"
done
model_tool --combine $models -o $output
popd
popd