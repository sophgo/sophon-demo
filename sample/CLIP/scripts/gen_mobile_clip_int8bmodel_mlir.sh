#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))

if [ ! $1 ]; then
    target=bm1684x
    target_dir=BM1684X
else
    target=${1,,}
    target_dir=${target^^}
    if test $target = "bm1684"
    then
        echo "bm1684 do not support fp16"
        exit
    fi
fi

outdir=../models/$target_dir
function gen_image_encode_mlir()
{
    model_transform.py \
      --model_name mobile_clip_image_b \
      --model_def ../models/onnx/mobile_clip_image_b.onnx \
      --input_shapes [[$1,3,224,224]] \
      --pixel_format rgb \
      --mlir mobile_clip_image_b_$1b.mlir
}

function gen_image_cali_table()
{
    run_calibration.py mobile_clip_image_b_$1b.mlir \
        --dataset ../datasets/cali_npz/image_mb \
        --sq \
        --input_num 100 \
        --cali_method mse \
        -o mb_clip_image_cali_table
}

function gen_image_encode_int8bmodel()
{
    model_deploy.py \
        --mlir mobile_clip_image_b_$1b.mlir \
        --calibration_table mb_clip_image_cali_table \
        --quantize_table mb_clip_image_qtable \
        --matmul_perchannel \
        --quantize INT8 \
        --chip $target \
        --model ./mobile_clip_image_b_${target}_int8_$1b.bmodel

    mv ./mobile_clip_image_b_${target}_int8_$1b.bmodel $outdir/

    if test $target = "bm1688";then
        model_deploy.py \
            --mlir mobile_clip_image_b_$1b.mlir \
            --calibration_table mb_clip_image_cali_table \
            --quantize_table mb_clip_image_qtable \
            --matmul_perchannel \
            --quantize INT8 \
            --chip $target \
            --model mobile_clip_image_b_${target}_int8_$1b_2core.bmodel \
            --num_core 2
        mv mobile_clip_image_b_${target}_int8_$1b_2core.bmodel $outdir/
    fi
}

function gen_text_encoder_mlir()
{
    model_transform.py \
      --model_name mobile_clip_text_b \
      --model_def ../models/onnx/mobile_clip_text_b.onnx \
      --input_shapes [[$1,77]] \
      --pixel_format rgb \
      --mlir mobile_clip_text_b_$1b.mlir
}

function gen_text_cali_table()
{
    run_calibration.py mobile_clip_text_b_$1b.mlir \
        --dataset ../datasets/cali_npz/text \
        --sq \
        --input_num 100 \
        --cali_method mse \
        -o mb_clip_text_cali_table
}
function gen_text_encoder_int8bmodel()
{
    model_deploy.py \
        --mlir mobile_clip_text_b_$1b.mlir \
        --quantize INT8 \
        --calibration_table mb_clip_text_cali_table \
        --quantize_table mb_clip_text_qtable \
        --matmul_perchannel \
        --chip $target \
        --model ./mobile_clip_text_b_${target}_int8_$1b.bmodel

    mv ./mobile_clip_text_b_${target}_int8_$1b.bmodel $outdir/

    if test $target = "bm1688";then
        model_deploy.py \
            --mlir mobile_clip_text_b_$1b.mlir \
            --quantize INT8 \
            --calibration_table mb_clip_text_cali_table \
            --quantize_table mb_clip_text_qtable \
            --matmul_perchannel \
            --chip $target \
            --model mobile_clip_text_b_${target}_int8_$1b_2core.bmodel \
            --num_core 2
        mv mobile_clip_text_b_${target}_int8_$1b_2core.bmodel $outdir/
    fi
}


pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi

# batch_size=1
# image encode
gen_image_encode_mlir 1
gen_image_cali_table 1
gen_image_encode_int8bmodel 1
# text encode
gen_text_encoder_mlir 1
gen_text_cali_table 1
gen_text_encoder_int8bmodel 1


# batch_size=4

# gen_image_encode_mlir 4
# gen_image_cali_table 4
# gen_image_encode_int8bmodel 4
# gen_text_encoder_mlir 4
# gen_text_cali_table 4
# gen_text_encoder_int8bmodel 4

popd
