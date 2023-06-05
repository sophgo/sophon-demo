#!/bin/bash
model_dir=$(cd `dirname $BASH_SOURCE[0]`/ && pwd)

echo "Do not support int8 now."
exit

if [ ! $1 ]; then
    target=bm1684x
    target_dir=BM1684X
else
    target=${1,,}
    target_dir=${target^^}
fi

outdir=../models/$target_dir

function calibration_det(){
    function gen_mlir_det()
    {
        model_transform.py \
            --model_name ch_PP-OCRv3_det \
            --model_def ../models/onnx/ch_PP-OCRv3_det.onnx \
            --input_shapes [[$1,3,640,640]] \
            --mlir ch_PP-OCRv3_det_$1b.mlir \
            --mean 123.675,116.28,103.53 \
            --scale 0.01712475,0.017507,0.0174292 \
            --keep_aspect_ratio \
            --pixel_format bgr
    }

    function gen_cali_table_det()
    {
        run_calibration.py ch_PP-OCRv3_det_$1b.mlir \
            --dataset ../datasets/cali_set_det \
            --input_num 128 \
            -o ppocrv3_det_cali_table
        
        run_qtable.py ch_PP-OCRv3_det_$1b.mlir \
            --dataset ../datasets/cali_set_det \
            --input_num 128 \
            --calibration_table ppocrv3_det_cali_table \
            --chip $target \
            -o det_qtable
    }

    function gen_int8bmodel_det()
    {
        model_deploy.py \
            --mlir ch_PP-OCRv3_det_$1b.mlir \
            --quantize INT8 \
            --chip $target \
            --calibration_table ppocrv3_det_cali_table \
            --quantize_table det_qtable \
            --model ch_PP-OCRv3_det_int8_$1b.bmodel

        mv ch_PP-OCRv3_det_int8_$1b.bmodel $outdir/
    }

    pushd $model_dir
    if [ ! -d $outdir ]; then
        mkdir -p $outdir
    fi

    echo -e "===================================="
    echo -e "Detection model calibration start!"
    # batch_size=1
    gen_mlir_det 1
    gen_cali_table_det 1
    gen_int8bmodel_det 1

    # batch_size=4
    gen_mlir_det 4
    gen_cali_table_det 4
    gen_int8bmodel_det 4
    
    model_tool --combine $outdir/ch_PP-OCRv3_det_int8_*.bmodel -o $outdir/ch_PP-OCRv3_det_int8.bmodel
    rm -r $outdir/ch_PP-OCRv3_det_int8_*.bmodel
    echo -e "Detection model calibration end!"
    echo -e "====================================\n"
}

function calibration_rec(){

    function gen_int8bmodel_rec()
    {
        model_transform.py \
            --model_name ch_PP-OCRv3_rec \
            --model_def ../models/onnx/ch_PP-OCRv3_rec.onnx \
            --input_shapes [[$1,3,48,$2]] \
            --mlir ch_PP-OCRv3_rec_$1b.mlir \
            --mean 127.5,127.5,127.5 \
            --scale 0.0078125,0.0078125,0.0078125 \
            --pixel_format bgr
        run_calibration.py ch_PP-OCRv3_rec_$1b.mlir \
            --dataset ../datasets/cali_set_rec \
            --input_num 128 \
            -o ppocrv3_rec_cali_table
        run_qtable.py ch_PP-OCRv3_rec_$1b.mlir \
            --dataset ../datasets/cali_set_rec \
            --input_num 128 \
            --calibration_table ppocrv3_rec_cali_table \
            --chip $target \
            -o rec_qtable
        model_deploy.py \
            --mlir ch_PP-OCRv3_rec_$1b.mlir \
            --quantize INT8 \
            --chip $target \
            --calibration_table ppocrv3_rec_cali_table \
            --quantize_table rec_qtable \
            --model ch_PP-OCRv3_rec_int8_$1b_$2.bmodel

        mv ch_PP-OCRv3_rec_int8_$1b_$2.bmodel $outdir/
    }


    echo -e "===================================="
    echo "Recognition model calibration start!"
    gen_int8bmodel_rec 1 320
    gen_int8bmodel_rec 1 640
    gen_int8bmodel_rec 4 320
    gen_int8bmodel_rec 4 640
    echo "Combining bmodels..."
    model_tool --combine $outdir/ch_PP-OCRv3_rec_int8_1b_320.bmodel \
                         $outdir/ch_PP-OCRv3_rec_int8_1b_640.bmodel \
                         $outdir/ch_PP-OCRv3_rec_int8_4b_320.bmodel \
                         $outdir/ch_PP-OCRv3_rec_int8_4b_640.bmodel -o $outdir/ch_PP-OCRv3_rec_int8.bmodel
    rm -r $outdir/ch_PP-OCRv3_rec_int8_*b_320.bmodel $outdir/ch_PP-OCRv3_rec_int8_*b_640.bmodel

    echo "Recognition model calibration end!"
    echo -e "====================================\n"
}


function calibration_cls(){
    function gen_int8bmodel_cls()
    {
        model_transform.py \
            --model_name ch_PP-OCRv3_cls \
            --model_def ../models/onnx/ch_PP-OCRv3_cls.onnx \
            --input_shapes [[$1,3,48,192]] \
            --mlir ch_PP-OCRv3_cls_$1b.mlir \
            --mean 127.5,127.5,127.5 \
            --scale 0.0078125,0.0078125,0.0078125 \
            --keep_aspect_ratio \
            --pixel_format bgr
        run_calibration.py ch_PP-OCRv3_cls_$1b.mlir \
            --dataset ../datasets/cali_set_rec \
            --input_num 128 \
            -o ppocrv3_cls_cali_table
        model_deploy.py \
            --mlir ch_PP-OCRv3_cls_$1b.mlir \
            --quantize INT8 \
            --chip $target \
            --calibration_table ppocrv3_cls_cali_table \
            --model ch_PP-OCRv3_cls_int8_$1b.bmodel

        mv ch_PP-OCRv3_cls_int8_$1b.bmodel $outdir/
    }


    echo -e "===================================="
    echo "Classification model calibration start!"
    gen_int8bmodel_cls 1
    gen_int8bmodel_cls 4
    model_tool --combine $outdir/ch_PP-OCRv3_cls_int8_*.bmodel -o $outdir/ch_PP-OCRv3_cls_int8.bmodel
    rm -r $outdir/ch_PP-OCRv3_cls_int8_*.bmodel
    echo "Classification model calibration end!"
    echo -e "====================================\n"
}

pushd $model_dir

calibration_det
calibration_rec
# calibration_cls

popd