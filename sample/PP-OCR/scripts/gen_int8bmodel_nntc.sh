#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))

echo "Do not support int8 now."
exit


if [ ! $1 ]; then
    target="BM1684"
else
    target=$1
fi
outdir=../models/$target
echo $outdir

mkdir -p $outdir

model_det_dir=../models/paddle/ch_PP-OCRv3_det_infer
model_cls_dir=../models/paddle/ch_ppocr_mobile_v2.0_cls_infer
model_rec_dir=../models/paddle/ch_PP-OCRv3_rec_infer

function gen_int8bmodel_det()
{
    python3 -m ufw.cali.cali_model --net_name=PP-OCRv3_det \
                        --input_shapes=[1,3,640,640] \
                        --model=$model_det_dir/inference.pdmodel \
                        --cali_image_path ../datasets/cali_set_det \
                        --cali_image_preprocess "resize_h=640,resize_w=640; \
                        mean_value=123.675:116.28:103.53,scale=0.01712475:0.017507:0.0174292,bgr2rgb=True" \
                        --cali_iterations 128 \
                        --target=$target \
                        --input_names="x" \
                        --output_names="sigmoid_0.tmp_0" \
                        --try_cali_accuracy_opt="-fpfwd_inputs=conv2d_211.tmp_0,-fpfwd_outputs=conv2d_267.tmp_0" \
                        --debug_cmd="not_suspend"
    mv $model_det_dir/PP-OCRv3_det_batch1/compilation.bmodel $outdir/ch_PP-OCRv3_det_int8_1b.bmodel
    
    bmnetu --model=$model_det_dir/PP-OCRv3_det_bmpaddle_deploy_int8_unique_top.prototxt  \
           --weight=$model_det_dir/PP-OCRv3_det_bmpaddle.int8umodel \
           -net_name=PP-OCRv3_det \
           --shapes=[4,3,640,640] \
           -target=$target \
           -opt=2
    mv compilation/compilation.bmodel $outdir/ch_PP-OCRv3_det_int8_4b.bmodel

    tpu_model --combine $outdir/ch_PP-OCRv3_det_int8_*.bmodel -o $outdir/ch_PP-OCRv3_det_int8.bmodel
    rm -r $outdir/ch_PP-OCRv3_det_int8_*.bmodel
}

function gen_int8bmodel_cls()
{
    python3 -m ufw.cali.cali_model --net_name=PP-OCRv3_cls \
                        --target=$target \
                        --input_shapes=[1,3,48,192] \
                        --model=$model_cls_dir/inference.pdmodel \
                        --cali_image_path ../datasets/cali_set_rec \
                        --cali_image_preprocess "resize_h=48,resize_w=192; \
                        mean_value=127.5:127.5:127.5,scale=0.0078125,bgr2rgb=True" \
                        --cali_iterations 128 \
                        --input_names="x" \
                        --output_names="softmax_0.tmp_0" \
                        --debug_cmd="not_suspend"
    mv $model_cls_dir/PP-OCRv3_cls_batch1/compilation.bmodel $outdir/ch_PP-OCRv3_cls_int8_1b.bmodel
    
    bmnetu --model=$model_cls_dir/PP-OCRv3_cls_bmpaddle_deploy_int8_unique_top.prototxt  \
           --weight=$model_cls_dir/PP-OCRv3_cls_bmpaddle.int8umodel \
           -net_name=PP-OCRv3_cls \
           --shapes=[4,3,48,192] \
           -target=$target \
           -opt=2
    mv compilation/compilation.bmodel $outdir/ch_PP-OCRv3_cls_int8_4b.bmodel
        
    tpu_model --combine $outdir/ch_PP-OCRv3_cls_int8_*.bmodel -o $outdir/ch_PP-OCRv3_cls_int8.bmodel
    rm -r $outdir/ch_PP-OCRv3_det_int8_*.bmodel
}

function gen_int8bmodel_rec()
{
    python3 -m ufw.cali.cali_model --net_name=PP-OCRv3_rec \
                        --target=$target \
                        --input_shapes=[1,3,48,$1] \
                        --model=$model_rec_dir/inference.pdmodel \
                        --cali_image_path ../datasets/cali_set_rec \
                        --cali_image_preprocess "resize_h=48,resize_w=$1; \
                        mean_value=127.5:127.5:127.5,scale=0.0078125,bgr2rgb=True" \
                        --cali_iterations 128 \
                        --input_names="x" \
                        --output_names="softmax_5.tmp_0" \
                        --try_cali_accuracy_opt="-fpfwd_inputs=conv2d_97.tmp_0,-fpfwd_outputs=conv2d_119.tmp_0" \
                        --debug_cmd="not_suspend"
    mv $model_rec_dir/PP-OCRv3_rec_batch1/compilation.bmodel $outdir/ch_PP-OCRv3_rec_int8_1b_$1.bmodel

}

pushd $model_dir


gen_int8bmodel_det 
gen_int8bmodel_cls 
gen_int8bmodel_rec 320
gen_int8bmodel_rec 640
tpu_model --combine $outdir/ch_PP-OCRv3_rec_int8_1b_320.bmodel \
                     $outdir/ch_PP-OCRv3_rec_int8_1b_640.bmodel \
                     $outdir/ch_PP-OCRv3_rec_int8_4b_320.bmodel \
                     $outdir/ch_PP-OCRv3_rec_int8_4b_640.bmodel -o $outdir/ch_PP-OCRv3_rec_int8.bmodel
rm -r $outdir/ch_PP-OCRv3_rec_int8_*b_320.bmodel $outdir/ch_PP-OCRv3_rec_int8_*b_640.bmodel
popd