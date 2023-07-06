#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))
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

function gen_fp32bmodel_det()
{
    python3 -m bmpaddle --net_name=PP-OCRv3_det \
                        --target=$target \
                        --opt=2 \
                        --cmp=true \
                        --shapes=[$1,3,640,640] \
                        --model=$model_det_dir \
                        --outdir=$outdir \
                        --dyn=false \
                        --output_names="sigmoid_0.tmp_0"
    mv $outdir/compilation.bmodel $outdir/ch_PP-OCRv3_det_fp32_$1b.bmodel
}

function gen_fp32bmodel_cls()
{
    python3 -m bmpaddle --net_name=PP-OCRv3_cls \
                        --target=$target \
                        --opt=2 \
                        --cmp=true \
                        --shapes=[$1,3,48,192] \
                        --model=$model_cls_dir \
                        --outdir=$outdir \
                        --dyn=false \
                        --output_names="softmax_0.tmp_0"
    mv $outdir/compilation.bmodel $outdir/ch_PP-OCRv3_cls_fp32_$1b.bmodel
}

function gen_fp32bmodel_rec()
{
    python3 -m bmpaddle --net_name=PP-OCRv3_rec \
                        --target=$target \
                        --opt=2 \
                        --cmp=true \
                        --shapes=[$1,3,48,$2] \
                        --model=$model_rec_dir \
                        --outdir=$outdir \
                        --dyn=false \
                        --output_names="softmax_5.tmp_0"
    mv $outdir/compilation.bmodel $outdir/ch_PP-OCRv3_rec_fp32_$1b_$2.bmodel
}

pushd $model_dir


gen_fp32bmodel_det 1
gen_fp32bmodel_cls 1
gen_fp32bmodel_rec 1 320
gen_fp32bmodel_rec 1 640
tpu_model --combine $outdir/ch_PP-OCRv3_rec_fp32_1b_320.bmodel $outdir/ch_PP-OCRv3_rec_fp32_1b_640.bmodel -o $outdir/ch_PP-OCRv3_rec_fp32_1b.bmodel
rm -r $outdir/ch_PP-OCRv3_rec_fp32_1b_320.bmodel $outdir/ch_PP-OCRv3_rec_fp32_1b_640.bmodel
popd