#!/bin/bash
set -x
# CV84X6(bm1684x2) FLUX.1-schnell bmodels: clip F16 + t5 head/tail BF16 + t5 blocks W4BF16
# + transformer head/tail BF16 + transformer blocks W4BF16 + tiny_vae(taef1) BF16
# 单芯运行; 文件切分/命名与 gen_bmodel.sh 一致, 以便 flux_pipeline 按名字加载
model_dir=$(dirname $(readlink -f "$0"))
pushd $model_dir

chip_type=bm1684x2
outdir=../models/CV84X6/
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi

function get_clip()
{
    mkdir -p ./clip
    pushd clip
    clip_onnx_pt_path=../../models/onnx_pt/clip/
    prefix=clip
    quant="F16"

    name=head
    shape=[[1,77]]
    model_transform.py --model_name $prefix'_'$name --input_shape $shape --model_def $clip_onnx_pt_path$prefix'_'$name.pt --mlir $name.mlir
    model_deploy.py --mlir $name.mlir --quantize $quant --chip $chip_type --model $prefix'_'$name'_'$quant.bmodel

    block_num=11
    for i in $(seq 0 $block_num);
    do
        name=block_$i
        shape=[[1,77,768]]
        model_transform.py --model_name $prefix'_'$name --input_shape $shape --model_def $clip_onnx_pt_path$prefix'_'$name.pt --mlir $name.mlir
        model_deploy.py --mlir $name.mlir --quantize $quant --chip $chip_type --model $prefix'_'$name'_'$quant.bmodel
    done

    name=tail
    shape=[[1,77,768],[1,77]]
    model_transform.py --model_name $prefix'_'$name --input_shape $shape --model_def $clip_onnx_pt_path$prefix'_'$name.pt --mlir $name.mlir
    model_deploy.py --mlir $name.mlir --quantize $quant --chip $chip_type --model $prefix'_'$name'_'$quant.bmodel

    files=$(ls *.bmodel | sort -V)
    files=$(echo "$files" | tr '\n' ' ')
    model_tool --combine $files -o ../../models/CV84X6/clip.bmodel
    popd
}

function get_t5()
{
    mkdir -p ./t5
    pushd t5
    prefix=t5
    t5_onnx_pt_path=../../models/onnx_pt/t5/

    name=head
    shape=[[1,512]]
    quant="F16"
    model_transform.py --model_name $prefix'_'$name --input_shape $shape --model_def $t5_onnx_pt_path$prefix'_'$name.onnx --mlir $name.mlir
    model_deploy.py --mlir $name.mlir --quantize $quant --chip $chip_type --model $prefix'_'$name'_'$quant.bmodel

    block_num=23
    quant="W4BF16"
    for i in $(seq 0 $block_num);
    do
        name=block_$i
        shape=[[1,512,4096]]
        model_transform.py --model_name $prefix'_'$name --input_shape $shape --model_def $t5_onnx_pt_path$prefix'_'$name.onnx --mlir $name.mlir
        model_deploy.py --mlir $name.mlir --quantize $quant --chip $chip_type --model $prefix'_'$name'_'$quant.bmodel
    done

    name=tail
    shape=[[1,512,4096]]
    quant="BF16"
    model_transform.py --model_name $prefix'_'$name --input_shape $shape --model_def $t5_onnx_pt_path$prefix'_'$name.pt --mlir $name.mlir
    model_deploy.py --mlir $name.mlir --quantize $quant --chip $chip_type --model $prefix'_'$name'_'$quant.bmodel

    files=$(ls *.bmodel | sort -V)
    files=$(echo "$files" | tr '\n' ' ')
    model_tool --combine $files -o ../../models/CV84X6/w4bf16_t5.bmodel
    popd
}

function get_transformer()
{
    mkdir -p schnell_W4BF16
    pushd schnell_W4BF16
    onnx_pt_path=../../models/onnx_pt/schnell_transformer/
    flux_type=schnell

    shape="[[1,4096,64],[1],[1,768],[1,512,4096]]"

    name=head
    quant="BF16"
    model_transform.py --model_name $flux_type"_"$name --input_shape $shape --model_def $onnx_pt_path$flux_type"_"$name.pt --mlir $name.mlir
    model_deploy.py --mlir $name.mlir --quantize $quant --chip $chip_type --model $name"_"$quant.bmodel

    quantize=W4BF16
    block_num=18
    for i in $(seq 0 $block_num);
    do
        name=trans_block_$i
        shape=[[1,4096,3072],[1,512,3072],[1,3072],[1,4608,1,64,2,2]]
        model_transform.py --model_name $flux_type'_'$name --input_shape $shape --model_def $onnx_pt_path$flux_type"_"$name.pt --mlir $name.mlir
        model_deploy.py --mlir $name.mlir --quantize $quantize --chip $chip_type --model $name'_'$quantize.bmodel
    done

    block_num=37
    for i in $(seq 0 $block_num);
    do
        name=single_trans_block_$i
        shape=[[1,4608,3072],[1,3072],[1,4608,1,64,2,2]]
        model_transform.py --model_name $flux_type'_'$name --input_shape $shape --model_def $onnx_pt_path$flux_type"_"$name.pt --mlir $name.mlir
        model_deploy.py --mlir $name.mlir --quantize $quantize --chip $chip_type --model $name'_'$quantize.bmodel
    done

    name=tail
    shape=[[1,4096,3072],[1,3072]]
    quant="BF16"
    model_transform.py --model_name $flux_type'_'$name --input_shape $shape --model_def $onnx_pt_path$flux_type"_"$name.pt --mlir $name.mlir
    model_deploy.py --mlir $name.mlir --quantize $quant --chip $chip_type --model $name"_"$quant.bmodel

    files=$(ls *.bmodel | sort -V)
    files=$(echo "$files" | tr '\n' ' ')
    model_tool --combine $files -o ../../models/CV84X6/schnell_w4bf16_transformer.bmodel
    popd
}

function get_vae()
{
    mkdir -p vae
    pushd vae
    onnx_pt_path=../../models/onnx_pt/vae/
    name=tiny_vae_decoder
    shape=[[1,16,128,128]]
    quant=BF16
    model_transform.py --model_name vae_decoder --input_shape $shape --model_def $onnx_pt_path$name.onnx --mlir $name.mlir
    model_deploy.py --mlir $name.mlir --quantize $quant --chip $chip_type --model $name'_bf16.bmodel'
    mv $name'_bf16.bmodel' ../../models/CV84X6/
    popd
}

get_clip
get_t5
get_transformer
get_vae

popd
