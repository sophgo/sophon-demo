#!/bin/bash
# CV84X6(bm1684x2) singlize bmodels: text_encoder F16 + unet/vae BF16
# 注意: CV84X6 当前固件 F32 的 matmul/fc 会命中 mm1 断言, text_encoder 只能编 F16;
#       unet/vae 为 BF16(与 BM1684X 的 F16 版本一致, 门外均为 fp32 接口)。
model_dir=$(dirname $(readlink -f "$0"))
outdir=../models/CV84X6/singlize/

if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi

function gen_text_encoder_mlir()
{
    model_transform.py \
        --model_name encoder \
        --model_def ../models/onnx_pt/text_encoder_1684x_f32.onnx \
        --input_shapes [[1,77]] \
        --mlir encoder.mlir
}

function gen_text_encoder_f16bmodel()
{
    model_deploy.py \
        --mlir encoder.mlir \
        --quantize F16 \
        --chip bm1684x2 \
        --model text_encoder_cv84x6_f16.bmodel

    mv text_encoder_cv84x6_f16.bmodel $outdir/
}

pushd $model_dir

gen_text_encoder_mlir
gen_text_encoder_f16bmodel

popd
