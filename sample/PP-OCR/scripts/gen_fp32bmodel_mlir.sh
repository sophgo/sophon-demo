#!/bin/bash
model_dir=$(cd `dirname $BASH_SOURCE[0]`/ && pwd)

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
        --model_name ch_PP-OCRv4_det \
        --model_def ../models/onnx/ch_PP-OCRv4_det.onnx \
        --input_shapes [[$1,3,640,640]] \
        --mlir ch_PP-OCRv4_det_$1b.mlir

    model_transform.py \
        --model_name ch_PP-OCRv4_rec \
        --model_def ../models/onnx/ch_PP-OCRv4_rec.onnx \
        --input_shapes [[$1,3,48,320]] \
        --mlir ch_PP-OCRv4_rec_$1b_320.mlir

    model_transform.py \
        --model_name ch_PP-OCRv4_rec \
        --model_def ../models/onnx/ch_PP-OCRv4_rec.onnx \
        --input_shapes [[$1,3,48,640]] \
        --mlir ch_PP-OCRv4_rec_$1b_640.mlir

    # model_transform.py \
    #     --model_name ch_PP-OCRv4_rec \
    #     --model_def ../models/onnx/ch_PP-OCRv4_rec.onnx \
    #     --input_shapes [[$1,3,48,960]] \
    #     --mlir ch_PP-OCRv4_rec_$1b_960.mlir
}


function gen_fp32bmodel()
{
    model_deploy.py \
        --mlir ch_PP-OCRv4_det_$1b.mlir \
        --quantize F32 \
        --chip $target \
        --model ch_PP-OCRv4_det_fp32_$1b.bmodel

    mv ch_PP-OCRv4_det_fp32_$1b.bmodel $outdir/

    model_deploy.py \
        --mlir ch_PP-OCRv4_rec_$1b_320.mlir \
        --quantize F32 \
        --chip $target \
        --model ch_PP-OCRv4_rec_fp32_$1b_320.bmodel

    mv ch_PP-OCRv4_rec_fp32_$1b_320.bmodel $outdir/

    model_deploy.py \
        --mlir ch_PP-OCRv4_rec_$1b_640.mlir \
        --quantize F32 \
        --chip $target \
        --model ch_PP-OCRv4_rec_fp32_$1b_640.bmodel

    mv ch_PP-OCRv4_rec_fp32_$1b_640.bmodel $outdir/

    # model_deploy.py \
    #     --mlir ch_PP-OCRv4_rec_$1b_960.mlir \
    #     --quantize F32 \
    #     --chip $target \
    #     --model ch_PP-OCRv4_rec_fp32_$1b_960.bmodel

    # mv ch_PP-OCRv4_rec_fp32_$1b_960.bmodel $outdir/
}

function gen_fp32bmodel_multicore()
{
    model_deploy.py \
        --mlir ch_PP-OCRv4_det_$1b.mlir \
        --quantize F32 \
        --chip $target \
        --num_core $2 \
        --model ch_PP-OCRv4_det_fp32_$1b_$2core.bmodel

    mv ch_PP-OCRv4_det_fp32_$1b_$2core.bmodel $outdir/

    model_deploy.py \
        --mlir ch_PP-OCRv4_rec_$1b_320.mlir \
        --quantize F32 \
        --chip $target \
        --num_core $2 \
        --model ch_PP-OCRv4_rec_fp32_$1b_320_$2core.bmodel

    mv ch_PP-OCRv4_rec_fp32_$1b_320_$2core.bmodel $outdir/

    model_deploy.py \
        --mlir ch_PP-OCRv4_rec_$1b_640.mlir \
        --quantize F32 \
        --chip $target \
        --num_core $2 \
        --model ch_PP-OCRv4_rec_fp32_$1b_640_$2core.bmodel

    mv ch_PP-OCRv4_rec_fp32_$1b_640_$2core.bmodel $outdir/
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
gen_mlir 1
gen_fp32bmodel 1

# batch_size=4
gen_mlir 4
gen_fp32bmodel 4
    
echo "Combining bmodels..."
model_tool --combine $outdir/ch_PP-OCRv4_det_fp32_*.bmodel -o $outdir/ch_PP-OCRv4_det_fp32.bmodel
rm -r $outdir/ch_PP-OCRv4_det_fp32_*.bmodel
model_tool --combine $outdir/ch_PP-OCRv4_rec_fp32_*.bmodel -o $outdir/ch_PP-OCRv4_rec_fp32.bmodel
rm -r $outdir/ch_PP-OCRv4_rec_fp32_*.bmodel

if test $target = "bm1688";then
    echo "Generating multicore models..."
    gen_fp32bmodel_multicore 1 2
    gen_fp32bmodel_multicore 4 2
    echo "Combining bmodels..."
    model_tool --combine $outdir/ch_PP-OCRv4_det_fp32_*b_*2core.bmodel -o $outdir/ch_PP-OCRv4_det_fp32_2core.bmodel
    rm -r $outdir/ch_PP-OCRv4_det_fp32_*b_*.bmodel
    model_tool --combine $outdir/ch_PP-OCRv4_rec_fp32_*b_*2core.bmodel -o $outdir/ch_PP-OCRv4_rec_fp32_2core.bmodel
    rm -r $outdir/ch_PP-OCRv4_rec_fp32_*b_*.bmodel
fi

popd