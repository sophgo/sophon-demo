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
    onnx_path="../models/onnx/dfine_n_coco.onnx"
    model_transform.py \
        --model_name ${model_name} \
        --model_def $onnx_path \
        --input_shapes [[$1,3,640,640],[$1,2]] \
        --mean 0.0,0.0,0.0 \
        --scale 0.0039216,0.0039216,0.0039216 \
        --mlir ${model_name}_$1b.mlir
        # --test_input ../datasets/test/dog.jpg \
        # --test_result ${model_name}_top_outputs.npz
}

function gen_f16bmodel()
{
    model_deploy.py \
        --mlir ${model_name}_$1b.mlir \
        --quantize F16 \
        --chip $target \
        --model ${model_name}_f16_$1b.bmodel
        # --test_input ../datasets/test/dog.jpg \
        # --test_reference ${model_name}_top_outputs.npz \
        # --tolerance 0.99,0.99 \
        # --compare_all

    mv ${model_name}_f16_$1b.bmodel $outdir/
    if test $target = "bm1688";then
        model_deploy.py \
            --mlir ${model_name}_$1b.mlir \
            --quantize F16 \
            --chip $target \
            --model ${model_name}_f16_$1b_2core.bmodel \
            --num_core 2
            # --test_input ${model_name}_in_f32.npz \
            # --test_reference ${model_name}_top_outputs.npz \
            # --compare_all

        mv ${model_name}_f16_$1b_2core.bmodel $outdir/
    fi
}


pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
model_name=dfine_n_coco
gen_mlir 1
gen_f16bmodel 1

# batch_size=4
# gen_mlir 4
# gen_f16bmodel 4

popd
