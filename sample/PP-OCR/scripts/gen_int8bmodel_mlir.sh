#!/bin/bash
model_dir=$(cd `dirname $BASH_SOURCE[0]`/ && pwd)

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


function gen_mlir()
{
    model_transform.py \
        --model_name ch_PP-OCRv4_det \
        --model_def ../models/onnx/ch_PP-OCRv4_det.onnx \
        --input_shapes [[$1,3,640,640]] \
        --mlir ch_PP-OCRv4_det_$1b.mlir

    if [ $1 -eq 1 ]; then
        run_calibration.py ch_PP-OCRv4_det_$1b.mlir \
            --dataset ../datasets/cali_npz_det/ \
            --input_num 128 \
            --cali_method kl \
            -o ch_PP-OCRv4_det_ctable
    fi

    model_transform.py \
        --model_name ch_PP-OCRv4_rec \
        --model_def ../models/onnx/ch_PP-OCRv4_rec.onnx \
        --input_shapes [[$1,3,48,320]] \
        --mlir ch_PP-OCRv4_rec_$1b_320.mlir

    if [ $1 -eq 1 ]; then
        run_calibration.py ch_PP-OCRv4_rec_$1b_320.mlir \
            --dataset ../datasets/cali_npz_rec320/ \
            --input_num 94 \
            --cali_method mse \
            -o ch_PP-OCRv4_rec_320_ctable
    fi

    model_transform.py \
        --model_name ch_PP-OCRv4_rec \
        --model_def ../models/onnx/ch_PP-OCRv4_rec.onnx \
        --input_shapes [[$1,3,48,640]] \
        --mlir ch_PP-OCRv4_rec_$1b_640.mlir

    if [ $1 -eq 1 ]; then
        run_calibration.py ch_PP-OCRv4_rec_$1b_640.mlir \
            --dataset ../datasets/cali_npz_rec640/ \
            --input_num 34 \
            --cali_method kl \
            -o ch_PP-OCRv4_rec_640_ctable
    fi
}


function gen_int8bmodel()
{
    model_deploy.py \
        --mlir ch_PP-OCRv4_det_$1b.mlir \
        --quantize INT8 \
        --chip $target \
        --calibration_table ch_PP-OCRv4_det_ctable \
        --quantize_table ch_PP-OCRv4_det_qtable \
        --model ch_PP-OCRv4_det_int8_$1b.bmodel

    mv ch_PP-OCRv4_det_int8_$1b.bmodel $outdir/

    model_deploy.py \
        --mlir ch_PP-OCRv4_rec_$1b_320.mlir \
        --quantize INT8 \
        --chip $target \
        --calibration_table ch_PP-OCRv4_rec_320_ctable \
        --quantize_table ch_PP-OCRv4_rec_320_qtable \
        --model ch_PP-OCRv4_rec_int8_$1b_320.bmodel

    mv ch_PP-OCRv4_rec_int8_$1b_320.bmodel $outdir/

    model_deploy.py \
        --mlir ch_PP-OCRv4_rec_$1b_640.mlir \
        --quantize INT8 \
        --chip $target \
        --calibration_table ch_PP-OCRv4_rec_640_ctable \
        --quantize_table ch_PP-OCRv4_rec_640_qtable \
        --model ch_PP-OCRv4_rec_int8_$1b_640.bmodel

    mv ch_PP-OCRv4_rec_int8_$1b_640.bmodel $outdir/

}
function gen_int8bmodel_multicore()
{
    model_deploy.py \
        --mlir ch_PP-OCRv4_det_$1b.mlir \
        --quantize INT8 \
        --chip $target \
        --num_core $2 \
        --calibration_table ch_PP-OCRv4_det_ctable \
        --quantize_table ch_PP-OCRv4_det_qtable \
        --model ch_PP-OCRv4_det_int8_$1b_$2core.bmodel

    mv ch_PP-OCRv4_det_int8_$1b_$2core.bmodel $outdir/

    model_deploy.py \
        --mlir ch_PP-OCRv4_rec_$1b_320.mlir \
        --quantize INT8 \
        --chip $target \
        --num_core $2 \
        --calibration_table ch_PP-OCRv4_rec_320_ctable \
        --quantize_table ch_PP-OCRv4_rec_320_qtable \
        --model ch_PP-OCRv4_rec_int8_$1b_320_$2core.bmodel

    mv ch_PP-OCRv4_rec_int8_$1b_320_$2core.bmodel $outdir/

    model_deploy.py \
        --mlir ch_PP-OCRv4_rec_$1b_640.mlir \
        --quantize INT8 \
        --chip $target \
        --num_core $2 \
        --calibration_table ch_PP-OCRv4_rec_640_ctable \
        --quantize_table ch_PP-OCRv4_rec_640_qtable \
        --model ch_PP-OCRv4_rec_int8_$1b_640_$2core.bmodel

    mv ch_PP-OCRv4_rec_int8_$1b_640_$2core.bmodel $outdir/
}


pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
gen_mlir 1
gen_int8bmodel 1

# batch_size=4
gen_mlir 4
gen_int8bmodel 4

echo "Combining bmodels..."
model_tool --combine $outdir/ch_PP-OCRv4_det_int8_*.bmodel -o $outdir/ch_PP-OCRv4_det_int8.bmodel
rm -r $outdir/ch_PP-OCRv4_det_int8_*.bmodel
model_tool --combine $outdir/ch_PP-OCRv4_rec_int8_*.bmodel -o $outdir/ch_PP-OCRv4_rec_int8.bmodel
rm -r $outdir/ch_PP-OCRv4_rec_int8_*.bmodel

if test $target = "bm1688";then
    echo "Generating multicore models..."
    gen_int8bmodel_multicore 1 2
    gen_int8bmodel_multicore 4 2
    echo "Combining bmodels..."
    model_tool --combine $outdir/ch_PP-OCRv4_det_int8_*b_*2core.bmodel -o $outdir/ch_PP-OCRv4_det_int8_2core.bmodel
    rm -r $outdir/ch_PP-OCRv4_det_int8_*b_*.bmodel
    model_tool --combine $outdir/ch_PP-OCRv4_rec_int8_*b_*2core.bmodel -o $outdir/ch_PP-OCRv4_rec_int8_2core.bmodel
    rm -r $outdir/ch_PP-OCRv4_rec_int8_*b_*.bmodel
fi


popd