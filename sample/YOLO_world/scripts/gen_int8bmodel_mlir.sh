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

fp_forward_chip=$target
if test $target = "bm1688"; then
    fp_forward_chip=bm1684x
fi

function gen_mlir()
{
   model_transform.py \
        --model_name yoloworld${opt} \
        --model_def ../models/onnx/yoloworld${opt}.onnx \
        --input_shapes [[$1,3,640,640],[1,80,512]] \
        --mean 0.0,0.0,0.0 \
        --scale 0.0039216,0.0039216,0.0039216 \
        --output_names output  \
        --keep_aspect_ratio \
        --pixel_format rgb  \
        --test_input ../models/onnx/coco128_npz/000000000009.npz  \
        --test_result yoloworld${opt}_top_outputs.npz \
        --mlir yoloworld${opt}_$1b.mlir
}

function gen_cali_table()
{
    run_calibration.py yoloworld${opt}_$1b.mlir \
        --dataset ../models/onnx/coco128_npz \
        --input_num 32 \
        -o yoloworld${opt}_cali_table
}

function gen_qtable()
{
    fp_forward.py yoloworld${opt}_$1b.mlir \
        --quantize INT8 \
        --chip $fp_forward_chip \
        --fpfwd_outputs /model.22/cv3.2/cv3.2.2/Conv_output_0_Conv,/model.22/cv3.2/cv3.2.1/conv/Conv_output_0_Conv,/model.22/cv3.2/cv3.2.0/conv/Conv_output_0_Conv,/model.18/cv1/conv/Conv_output_0_Conv,/model.22/cv2.2/cv2.2.0/conv/Conv_output_0_Conv,/model.22/cv2.2/cv2.2.1/conv/Conv_output_0_Conv \
        -o yoloworld${opt}_qtable
}

function gen_int8bmodel()
{
    qtable_path=../models/onnx/yoloworld${opt}_qtable_fp16
    if test $target = "bm1684";then
        qtable_path=../models/onnx/yoloworld${opt}_qtable_fp32
    fi
    model_deploy.py \
        --mlir yoloworld${opt}_$1b.mlir \
        --quantize INT8 \
        --chip $target \
        --calibration_table yoloworld${opt}_cali_table \
        --quantize_table yoloworld${opt}_qtable \
        --test_input yoloworld${opt}_in_f32.npz \
        --test_reference yoloworld${opt}_top_outputs.npz \
        --model yoloworld${opt}_int8_$1b.bmodel

    mv yoloworld${opt}_int8_$1b.bmodel $outdir/
    if test $target = "bm1688";then
        model_deploy.py \
            --mlir yoloworld${opt}_$1b.mlir \
            --quantize INT8 \
            --chip $target \
            --model yoloworld${opt}_int8_$1b_2core.bmodel \
            --calibration_table yoloworld${opt}_cali_table \
            --num_core 2 \
            --quantize_table yoloworld${opt}_qtable \
            --test_input yoloworld${opt}_in_f32.npz \
            --test_reference yoloworld${opt}_top_outputs.npz 


        mv yoloworld${opt}_int8_$1b_2core.bmodel $outdir/
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
gen_cali_table 1
gen_qtable 1
gen_int8bmodel 1

# batch_size=1
# text encode
gen_text_encoder_mlir 1
gen_text_encoder_fp16bmodel 1

# batch_size=4
# gen_mlir 4
# gen_int8bmodel 4

# opt="_opt"
# # batch_size=1
# gen_mlir 1
# gen_cali_table 1
# gen_int8bmodel 1

# # batch_size=4
# gen_mlir 4
# gen_int8bmodel 4

popd