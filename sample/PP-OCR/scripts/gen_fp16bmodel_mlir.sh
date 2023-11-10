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
        --model_name ch_PP-OCRv3_det \
        --model_def ../models/onnx/ch_PP-OCRv3_det.onnx \
        --input_shapes [[$1,3,640,640]] \
        --mlir ch_PP-OCRv3_det_$1b.mlir
    model_transform.py \
        --model_name ch_PP-OCRv3_cls \
        --model_def ../models/onnx/ch_PP-OCRv3_cls.onnx \
        --input_shapes [[$1,3,48,192]] \
        --mlir ch_PP-OCRv3_cls_$1b.mlir
    model_transform.py \
        --model_name ch_PP-OCRv3_rec \
        --model_def ../models/onnx/ch_PP-OCRv3_rec.onnx \
        --input_shapes [[$1,3,48,320]] \
        --mlir ch_PP-OCRv3_rec_$1b_320.mlir
    model_transform.py \
        --model_name ch_PP-OCRv3_rec \
        --model_def ../models/onnx/ch_PP-OCRv3_rec.onnx \
        --input_shapes [[$1,3,48,640]] \
        --mlir ch_PP-OCRv3_rec_$1b_640.mlir
}


function gen_fp16bmodel()
{
    model_deploy.py \
        --mlir ch_PP-OCRv3_det_$1b.mlir \
        --quantize F16 \
        --chip $target \
        --model ch_PP-OCRv3_det_fp16_$1b.bmodel

    mv ch_PP-OCRv3_det_fp16_$1b.bmodel $outdir/
    
    model_deploy.py \
        --mlir ch_PP-OCRv3_cls_$1b.mlir \
        --quantize F16 \
        --chip $target \
        --model ch_PP-OCRv3_cls_fp16_$1b.bmodel

    mv ch_PP-OCRv3_cls_fp16_$1b.bmodel $outdir/

    model_deploy.py \
        --mlir ch_PP-OCRv3_rec_$1b_320.mlir \
        --quantize F16 \
        --chip $target \
        --model ch_PP-OCRv3_rec_fp16_$1b_320.bmodel

    mv ch_PP-OCRv3_rec_fp16_$1b_320.bmodel $outdir/

    model_deploy.py \
        --mlir ch_PP-OCRv3_rec_$1b_640.mlir \
        --quantize F16 \
        --chip $target \
        --model ch_PP-OCRv3_rec_fp16_$1b_640.bmodel

    mv ch_PP-OCRv3_rec_fp16_$1b_640.bmodel $outdir/

}
function gen_fp16bmodel_multicore()
{
    model_deploy.py \
        --mlir ch_PP-OCRv3_det_$1b.mlir \
        --quantize F16 \
        --chip $target \
        --num_core $2 \
        --model ch_PP-OCRv3_det_fp16_$1b_$2core.bmodel

    mv ch_PP-OCRv3_det_fp16_$1b_$2core.bmodel $outdir/
    
    model_deploy.py \
        --mlir ch_PP-OCRv3_cls_$1b.mlir \
        --quantize F16 \
        --chip $target \
        --num_core $2 \
        --model ch_PP-OCRv3_cls_fp16_$1b_$2core.bmodel

    mv ch_PP-OCRv3_cls_fp16_$1b_$2core.bmodel $outdir/

    model_deploy.py \
        --mlir ch_PP-OCRv3_rec_$1b_320.mlir \
        --quantize F16 \
        --chip $target \
        --num_core $2 \
        --model ch_PP-OCRv3_rec_fp16_$1b_320_$2core.bmodel

    mv ch_PP-OCRv3_rec_fp16_$1b_320_$2core.bmodel $outdir/

    model_deploy.py \
        --mlir ch_PP-OCRv3_rec_$1b_640.mlir \
        --quantize F16 \
        --chip $target \
        --num_core $2 \
        --model ch_PP-OCRv3_rec_fp16_$1b_640_$2core.bmodel

    mv ch_PP-OCRv3_rec_fp16_$1b_640_$2core.bmodel $outdir/
}


pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
gen_mlir 1
gen_fp16bmodel 1

# batch_size=4
gen_mlir 4
gen_fp16bmodel 4

echo "Combining bmodels..."
model_tool --combine $outdir/ch_PP-OCRv3_det_fp16_*.bmodel -o $outdir/ch_PP-OCRv3_det_fp16.bmodel
rm -r $outdir/ch_PP-OCRv3_det_fp16_*.bmodel
model_tool --combine $outdir/ch_PP-OCRv3_cls_fp16_*.bmodel -o $outdir/ch_PP-OCRv3_cls_fp16.bmodel
rm -r $outdir/ch_PP-OCRv3_cls_fp16_*.bmodel
model_tool --combine $outdir/ch_PP-OCRv3_rec_fp16_*.bmodel -o $outdir/ch_PP-OCRv3_rec_fp16.bmodel
rm -r $outdir/ch_PP-OCRv3_rec_fp16_*.bmodel

# if test $target = "bm1688";then
#     echo "Generating multicore models..."
#     gen_fp16bmodel_multicore 1 2
#     gen_fp16bmodel_multicore 4 2
#     echo "Combining bmodels..."
#     model_tool --combine $outdir/ch_PP-OCRv3_det_fp16_*b_*2core.bmodel -o $outdir/ch_PP-OCRv3_det_fp16_2core.bmodel
#     rm -r $outdir/ch_PP-OCRv3_det_fp16_*b_*.bmodel
#     model_tool --combine $outdir/ch_PP-OCRv3_cls_fp16_*b_*2core.bmodel -o $outdir/ch_PP-OCRv3_cls_fp16_2core.bmodel
#     rm -r $outdir/ch_PP-OCRv3_cls_fp16_*b_*.bmodel
#     model_tool --combine $outdir/ch_PP-OCRv3_rec_fp16_*b_*2core.bmodel -o $outdir/ch_PP-OCRv3_rec_fp16_2core.bmodel
#     rm -r $outdir/ch_PP-OCRv3_rec_fp16_*b_*.bmodel
# fi


popd