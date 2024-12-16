#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))

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
        --model_name yoloworld${opt} \
        --model_def ../models/onnx/yoloworld${opt}.onnx \
        --input_shapes [[$1,3,640,640],[1,80,512]] \
        --mean 0.0,0.0,0.0 \
        --scale 0.0039216,0.0039216,0.0039216 \
        --output_names output    \
        --keep_aspect_ratio \
        --pixel_format rgb  \
        --mlir yoloworld${opt}_$1b.mlir 
}

function gen_fp32bmodel()
{
    model_deploy.py \
        --mlir yoloworld${opt}_$1b.mlir \
        --quantize F32 \
        --chip $target \
        --model yoloworld${opt}_fp32_$1b.bmodel 

    mv yoloworld${opt}_fp32_$1b.bmodel $outdir/
    if test $target = "bm1688";then
        model_deploy.py \
            --mlir yoloworld${opt}_$1b.mlir \
            --quantize F32 \
            --chip $target \
            --model yoloworld${opt}_fp32_$1b_2core.bmodel \
            --num_core 2
            # --test_input ../datasets/test/3.jpg \
            # --test_reference yolov5_top.npz \
            # --debug 
        mv yoloworld${opt}_fp32_$1b_2core.bmodel $outdir/
    fi
}

function gen_text_encoder_mlir()
{
    model_transform.py \
      --model_name clip_text_vitb32 \
      --model_def ../models/onnx/clip_text_vitb32.onnx \
      --input_shapes [[$1,77]] \
      --pixel_format rgb \
      --mlir clip_text_vitb32_$1b.mlir
}

function gen_text_encoder_fp16bmodel()
{
    model_deploy.py \
        --mlir clip_text_vitb32_$1b.mlir \
        --quantize F16 \
        --chip $target \
        --model ./clip_text_vitb32_${target}_f16_$1b.bmodel

    mv ./clip_text_vitb32_${target}_f16_$1b.bmodel $outdir/
    if test $target = "bm1688";then
        model_deploy.py \
            --mlir clip_text_vitb32_$1b.mlir \
            --quantize F16 \
            --chip $target \
            --model clip_text_vitb32_${target}_f16_$1b_2core.bmodel \
            --num_core 2
        mv clip_text_vitb32_${target}_f16_$1b_2core.bmodel $outdir/
    fi
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
gen_mlir 1
gen_fp32bmodel 1

# batch_size=1
# text encode
gen_text_encoder_mlir 1
gen_text_encoder_fp16bmodel 1

# opt="_opt"
# gen_mlir 1
# gen_fp32bmodel 1
popd