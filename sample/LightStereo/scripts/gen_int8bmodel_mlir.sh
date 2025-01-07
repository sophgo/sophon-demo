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
        --model_name LightStereo \
        --model_def ../models/onnx/LightStereo-S-SceneFlow.onnx \
        --input_shapes [[$1,3,384,1248],[$1,3,384,1248]] \
        --mlir LightStereo-S-SceneFlow_$1b.mlir
        # --test_input ../datasets/cali_data/1.npz \
        # --test_result LightStereo_top_outputs.npz
        # test_input is only for 1b
}

function gen_cali_table()
{
    run_calibration.py LightStereo-S-SceneFlow_$1b.mlir \
        --dataset ../datasets/cali_data \
        --input_num 32 \
        -o LightStereo_cali_table \
        --search search_qtable \
        --quantize_table lightstereo_sensitive_layer
}

function gen_int8bmodel()
{
    model_deploy.py \
        --mlir LightStereo-S-SceneFlow_$1b.mlir \
        --quantize INT8 \
        --chip $target \
        --calibration_table LightStereo_cali_table \
        --model LightStereo-S-SceneFlow_int8_$1b.bmodel \
        --quant_input
        # --test_input ../datasets/cali_data/1.npz \
        # --test_reference LightStereo_top_outputs.npz \
        # --compare_all

    mv LightStereo-S-SceneFlow_int8_$1b.bmodel $outdir/
    if test $target = "bm1688";then
        model_deploy.py \
            --mlir LightStereo-S-SceneFlow_$1b.mlir \
            --quantize INT8 \
            --chip $target \
            --calibration_table LightStereo_cali_table \
            --model LightStereo-S-SceneFlow_int8_$1b_2core.bmodel \
            --quant_input \
            --num_core 2
            # --test_input ../datasets/cali_data/1.npz \
            # --test_reference LightStereo_top_outputs.npz

        mv LightStereo-S-SceneFlow_int8_$1b_2core.bmodel $outdir/
    fi
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
gen_mlir 1
gen_cali_table 1
gen_int8bmodel 1

# batch_size=4
gen_mlir 4
gen_int8bmodel 4

popd
