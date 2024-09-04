#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))
name=blip_vqa
if [ ! $1 ]; then
    target=bm1684x
    target_dir=BM1684X
else
    target=${1,,}
    target_dir=${target^^}
fi

outdir=../models/$target_dir
function gen_mlir()
{
    model_transform.py \
        --model_name ${name}_venc \
        --model_def ../models/onnx/${name}_venc.onnx \
        --input_shapes [[$1,3,480,480]] \
        --mlir ${name}_venc_$1b.mlir

    model_transform.py \
        --model_name ${name}_tenc \
        --model_def ../models/onnx/${name}_tenc.onnx \
        --input_shapes [[$1,901,768],[$1,35],[$1,35]] \
        --mlir ${name}_tenc_$1b.mlir

    model_transform.py \
        --model_name ${name}_tdec \
        --model_def ../models/onnx/${name}_tdec.onnx \
        --input_shapes [[$1,35,768]] \
        --mlir ${name}_tdec_$1b.mlir
}

function gen_fp32bmodel()
{
    model_deploy.py \
        --mlir ${name}_venc_$1b.mlir \
        --quantize F32 \
        --chip ${target} \
        --model ${name}_venc_${target}_f32_$1b.bmodel
    mv ./${name}_venc_${target}_f32_$1b.bmodel $outdir

    model_deploy.py \
        --mlir ${name}_tenc_$1b.mlir \
        --quantize F32 \
        --chip ${target} \
        --model ${name}_tenc_${target}_f32_$1b.bmodel
    mv ./${name}_tenc_${target}_f32_$1b.bmodel $outdir

    model_deploy.py \
        --mlir ${name}_tdec_$1b.mlir \
        --quantize F32 \
        --chip ${target} \
        --model ${name}_tdec_${target}_f32_$1b.bmodel
    mv ./${name}_tdec_${target}_f32_$1b.bmodel $outdir
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi

# batch_size=1
gen_mlir 1
gen_fp32bmodel 1

popd
