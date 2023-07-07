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
        --model_name c3d \
        --model_def ../models/onnx/c3d_ucf101.onnx \
        --input_shapes [[$1,3,16,112,112]] \
        --mlir c3d_$1b.mlir
}

function gen_cali_table()
{
    run_calibration.py c3d_$1b.mlir \
        --dataset ../datasets/cali_set_npy \
        --input_num 128 \
        -o c3d_cali_table
}

function gen_int8bmodel()
{
    model_deploy.py \
        --mlir c3d_$1b.mlir \
        --quantize INT8 \
        --chip $target \
        --calibration_table c3d_cali_table \
        --model $outdir/c3d_int8_$1b.bmodel
}

pushd $model_dir
if [ ! -d "$outdir" ]; then
    echo $pwd
    mkdir $outdir
fi

cd ../tools/
python3 c3d_npy.py
cd ../scripts

# batch_size=1
gen_mlir 1
gen_cali_table 1
gen_int8bmodel 1

# batch_size=4
gen_mlir 4
gen_int8bmodel 4
rm -r c3d* final_opt.onnx
popd