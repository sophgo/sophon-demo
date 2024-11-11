#!/bin/bash
scripts_dir=$(dirname $(readlink -f "$0"))
pushd $scripts_dir

set -ex
# 检查是否传入了额外的路径参数
if [ "$#" -ne 1 ]; then
    echo "Usage: \$0 <path>"
    exit 1
fi

if [ ! $1 ]; then
    target=bm1684x
    target_dir=BM1684X
else
    target=${1,,}
    target_dir=${target^^}
fi

outdir=../models/$target_dir
compile_dir=../models/compile_dir

pushd $scripts_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi

# 检查目标目录是否存在
if [ ! -d $compile_dir ]; then
    echo "Directory '$compile_dir' does not exist. Creating it now..."
    mkdir -p $compile_dir
    echo "Directory '$compile_dir' created successfully."
else
    echo "Directory '$compile_dir' already exists."
fi

# encoder1
function gen_encoder1()
{
    model_transform.py \
        --model_name audio_encoder_1 \
        --model_def ../onnx/audio_encoder_1.onnx \
        --test_input ../testInput/input_encoder_1.npz \
        --test_result audio_encoder_1_top_output.npz \
        --input_shapes [1,262144] \
        --mlir audio_encoder_1.mlir \
        --dynamic \
        --debug

    model_deploy.py \
        --mlir audio_encoder_1.mlir \
        --quantize F32 \
        --test_input ../testInput/input_encoder_1.npz \
        --test_reference audio_encoder_1_top_output.npz \
        --chip $target \
        --model audio_encoder_1.bmodel \
        --disable_layer_group \
        --compare_all \
    --dynamic
}

# encoder2
function gen_encoder2()
{
    model_transform.py \
        --model_name audio_encoder_2 \
        --model_def ../onnx/audio_encoder_2.onnx \
        --test_input ../testInput/input_encoder_2.npz \
        --test_result audio_encoder_2_top_output.npz \
        --input_shapes [[1,490,512],[1,8]] \
        --mlir audio_encoder_2.mlir \
        --dynamic \
        --debug

    model_deploy.py \
        --mlir audio_encoder_2.mlir \
        --quantize F32 \
        --test_input ../testInput/input_encoder_2.npz \
        --test_reference audio_encoder_2_top_output.npz \
        --chip $target \
        --model audio_encoder_2.bmodel \
        --disable_layer_group \
        --compare_all \
        --dynamic
}

# ppe
function gen_ppe()
{
    model_transform.py \
        --model_name ppe \
        --model_def ../onnx/ppe.onnx \
        --test_input ../testInput/input_ppe.npz \
        --test_result ppe_top_output.npz \
        --input_shapes [[1,490,64]] \
        --mlir ppe.mlir \
        --dynamic

    model_deploy.py \
        --mlir ppe.mlir \
        --quantize F16 \
        --test_input ../testInput/input_ppe.npz \
        --test_reference ppe_top_output.npz \
        --chip $target \
        --model ppe.bmodel \
        --compare_all \
        --dynamic
}
# decoder
function gen_decoder()
{
    model_transform.py \
        --model_name decoder \
        --model_def ../onnx/decoder.onnx \
        --test_input ../testInput/input_decoder.npz \
        --test_result decoder_top_output.npz \
        --input_shapes [[1,490,64],[1,490,64],[490,490]] \
        --mlir decoder.mlir \
        --dynamic

    model_deploy.py \
        --mlir decoder.mlir \
        --quantize F16 \
        --test_input ../testInput/input_decoder.npz \
        --test_reference decoder_top_output.npz \
        --chip $target \
        --model decoder_f16.bmodel \
        --compare_all \
        --dynamic
}
pushd $compile_dir
gen_encoder1
gen_encoder2
gen_ppe
gen_decoder
model_tool --combine decoder_f16.bmodel audio_encoder_1.bmodel audio_encoder_2.bmodel ppe.bmodel -o faceformer_f32.bmodel
mv faceformer_f32.bmodel ../$target_dir
popd
popd
popd