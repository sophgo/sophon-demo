#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))
if [ ! $1 ]; then
    target="BM1684X"
else
    target=$1
fi
outdir=../data/models/$target
echo $outdir

mkdir -p $outdir

model_det_dir=../data/models/paddle/ch_PP-OCRv2_det_infer
model_cls_dir=../data/models/paddle/ch_ppocr_mobile_v2.0_cls_infer
model_rec_dir=../data/models/paddle/ch_PP-OCRv2_rec_infer

function gen_fp32bmodel_det()
{
    python3 -m bmpaddle --net_name=PP-OCRv2_det \
                        --target=$target \
                        --opt=2 \
                        --cmp=true \
                        --shapes=[$1,3,960,960] \
                        --model=$model_det_dir \
                        --outdir=$outdir \
                        --dyn=false \
                        --output_names="save_infer_model/scale_0.tmp_1"
    mv $outdir/compilation.bmodel $outdir/ch_PP-OCRv2_det_$1b.bmodel

}

function gen_fp32bmodel_cls()
{
    python3 -m bmpaddle --net_name=ppocr_mobile_v2.0_cls \
                        --target=$target \
                        --opt=1 \
                        --cmp=true \
                        --shapes=[$1,3,48,192] \
                        --model=$model_cls_dir \
                        --outdir=$outdir \
                        --dyn=false \
                        --output_names="save_infer_model/scale_0.tmp_1"
    mv $outdir/compilation.bmodel $outdir/ch_ppocr_mobile_v2.0_cls_$1b.bmodel
}

function gen_fp32bmodel_rec()
{
    python3 -m bmpaddle --net_name=PP-OCRv2_rec \
                        --target=$target \
                        --opt=1 \
                        --cmp=true \
                        --shapes=[$1,3,32,$2] \
                        --model=$model_rec_dir \
                        --outdir=$outdir \
                        --dyn=false \
                        --output_names="save_infer_model/scale_0.tmp_1"
    mv $outdir/compilation.bmodel $outdir/ch_PP-OCRv2_rec_$2_$1b.bmodel
}

pushd $model_dir

if [ $1 = "BM1684" ]; then
    gen_fp32bmodel_det 1
    gen_fp32bmodel_det 4
    gen_fp32bmodel_cls 1
    gen_fp32bmodel_cls 4
    gen_fp32bmodel_rec 1 320
    gen_fp32bmodel_rec 4 320
    gen_fp32bmodel_rec 1 640
    gen_fp32bmodel_rec 4 640
    gen_fp32bmodel_rec 1 1280
    tpu_model --combine $outdir/ch_PP-OCRv2_det_1b.bmodel $outdir/ch_PP-OCRv2_det_4b.bmodel -o $outdir/ch_PP-OCRv2_det_fp32_b1b4.bmodel
    tpu_model --combine $outdir/ch_ppocr_mobile_v2.0_cls_1b.bmodel $outdir/ch_ppocr_mobile_v2.0_cls_4b.bmodel -o $outdir/ch_ppocr_mobile_v2.0_cls_fp32_b1b4.bmodel
    tpu_model --combine $outdir/ch_PP-OCRv2_rec_320_1b.bmodel $outdir/ch_PP-OCRv2_rec_320_4b.bmodel $outdir/ch_PP-OCRv2_rec_640_1b.bmodel $outdir/ch_PP-OCRv2_rec_640_4b.bmodel $outdir/ch_PP-OCRv2_rec_1280_1b.bmodel -o $outdir/ch_PP-OCRv2_rec_fp32_b1b4.bmodel
else
    gen_fp32bmodel_det 1
    # gen_fp32bmodel_det 4
    gen_fp32bmodel_cls 1
    # gen_fp32bmodel_cls 4
    gen_fp32bmodel_rec 1 320
    # gen_fp32bmodel_rec 4 320
    gen_fp32bmodel_rec 1 640
    # gen_fp32bmodel_rec 4 640
    gen_fp32bmodel_rec 1 1280
    # tpu_model --combine $outdir/ch_PP-OCRv2_det_1b.bmodel $outdir/ch_PP-OCRv2_det_4b.bmodel -o $outdir/ch_PP-OCRv2_det_fp32_b1b4.bmodel
    # tpu_model --combine $outdir/ch_ppocr_mobile_v2.0_cls_1b.bmodel $outdir/ch_ppocr_mobile_v2.0_cls_4b.bmodel -o $outdir/ch_ppocr_mobile_v2.0_cls_fp32_b1b4.bmodel
    tpu_model --combine $outdir/ch_PP-OCRv2_rec_320_1b.bmodel $outdir/ch_PP-OCRv2_rec_640_1b.bmodel $outdir/ch_PP-OCRv2_rec_1280_1b.bmodel -o $outdir/ch_PP-OCRv2_rec_fp32_b1.bmodel
fi

popd