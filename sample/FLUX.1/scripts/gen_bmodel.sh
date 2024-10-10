#!/bin/bash
set -x
model_dir=$(dirname $(readlink -f "$0"))
pushd $model_dir

flux_type="dev"
quantize="BF16"
chip_type=bm1684x
use_taef1=0

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in 
    --flux_type)
        flux_type="$2"
        shift 2
        ;;
    --quantize)
        quantize="$2"
        shift 2
        ;;
    --chip_type)
        chip_type="$2"
        shift 2
        ;;
    --use_taef1)
        use_taef1="$2"
        shift 2
        ;;
    *)
        shift
        ;;
    esac
done

chip_type_upper=$(echo "$chip_type" | tr '[:lower:]' '[:upper:]')

outdir=../models/$chip_type_upper/
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi

function get_clip()
{
    mkdir -p ./clip 
    pushd clip
    clip_onnx_pt_path=../../models/onnx_pt/clip/
    prefix=clip
    name=head
    shape=[[1,77]]
    quant="F16"
    model_transform.py --model_name $prefix'_'$name --input_shape $shape --model_def $clip_onnx_pt_path$prefix'_'$name.pt --mlir $name.mlir
    model_deploy.py --mlir $name.mlir --quantize $quant --chip $chip_type --model $prefix'_'$name'_'$quant.bmodel

    block_num=11
    for i in $(seq 0 $block_num);
    do
        name=block_$i
        shape=[[1,77,768]]
        quant="F16"
        model_transform.py --model_name $prefix'_'$name --input_shape $shape --model_def $clip_onnx_pt_path$prefix'_'$name.pt --mlir $name.mlir
        model_deploy.py --mlir $name.mlir --quantize $quant --chip $chip_type --model $prefix'_'$name'_'$quant.bmodel
    done

    name=tail
    shape=[[1,77,768],[1,77]]
    quant="F16"
    model_transform.py --model_name $prefix'_'$name --input_shape $shape --model_def $clip_onnx_pt_path$prefix'_'$name.pt --mlir $name.mlir
    model_deploy.py --mlir $name.mlir --quantize $quant --chip $chip_type --model $prefix'_'$name'_'$quant.bmodel

    files=$(ls *.bmodel | sort -V)
    files=$(echo "$files" | tr '\n' ' ')
    model_tool --combine $files -o ../../models/$chip_type_upper/clip.bmodel 
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
    model_tool --combine $files -o ../../models/$chip_type_upper/w4bf16_t5.bmodel 

    popd
}

function get_transformer()
{
    mkdir -p $flux_type'_'$quantize
    pushd $flux_type'_'$quantize
    onnx_pt_path=../../models/onnx_pt/$flux_type'_transformer'/
    if [ "$flux_type" == "schnell" ]; then
        shape="[[1,4096,64],[1],[1,768],[1,512,4096]]"
    elif [ "$flux_type" == "dev" ]; then
        shape="[[1,4096,64],[1],[1],[1,768],[1,512,4096]]"
    else
        echo "Invalid flux_type value. Please set it to 'schnell' or 'dev'."
        exit 1
    fi

    name=head
    quant="BF16"
    model_transform.py --model_name $flux_type"_"$name --input_shape $shape --model_def $onnx_pt_path$flux_type"_"$name.pt --mlir $name.mlir
    model_deploy.py --mlir $name.mlir --quantize $quant --chip $chip_type --model $name"_"$quant.bmodel 

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

    if [ $quantize == "BF16" ]; then
        name=""
        for i in $(seq 0 12);
        do
            name=$name"trans_block_"$i"_"$quantize".bmodel "
        done
        head="head_"$quant.bmodel
        name=$name" "$head

        model_tool --combine $name -o ../../models/$chip_type_upper/${flux_type}_bf16_transformer_on_device0.bmodel

        name=""

        for i in $(seq 13 18);
        do
            name=$name"trans_block_"$i"_"$quantize".bmodel "
        done

        for i in $(seq 0 27);
        do
            name=$name"single_trans_block_"$i"_"$quantize".bmodel "
        done

        model_tool --combine $name -o ../../models/$chip_type_upper/${flux_type}_bf16_transformer_on_device1.bmodel

        name=""
        for i in $(seq 28 37);
        do
            name=$name"single_trans_block_"$i"_"$quantize".bmodel "
        done
        tail="tail_"$quant.bmodel
        name=$name" "$tail
        model_tool --combine $name -o ../../models/$chip_type_upper/${flux_type}_bf16_transformer_on_device2.bmodel

    elif [ $quantize == "W4BF16" ]; then
        files=$(ls *.bmodel | sort -V)
        files=$(echo "$files" | tr '\n' ' ')
        model_tool --combine $files -o ../../models/$chip_type_upper/$flux_type'_w4bf16_transformer.bmodel' 
    fi

    popd
}


function get_vae()
{
    mkdir -p vae
    pushd vae
    onnx_pt_path=../../models/onnx_pt/vae/
    if [ $use_taef1 -eq 1 ]; then
        name=tiny_vae_decoder
    else
        name=vae_decoder
    fi
    shape=[[1,16,128,128]]
    quant=BF16
    model_transform.py --model_name $name --input_shape $shape --model_def $onnx_pt_path$name.onnx --mlir $name.mlir
    model_deploy.py --mlir $name.mlir --quantize $quant --chip $chip_type --model $name'_bf16.bmodel'
    mv $name'_bf16.bmodel' ../../models/$chip_type_upper/
    popd
}

get_clip
get_t5
get_transformer
get_vae

popd