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

# 注意：GLC 的 INT8 校准代价很高，实测解码器单个样本需 87 分钟激活采集
# 加 625 个算子的逐算子调优（慢算子约 26 s/it），编码器更慢；TPU-MLIR 官方
# 建议 100+ 样本。详见 ../README.md 的「8. FAQ」Q9。
# 可用 SEARCH 环境变量换更快的搜索策略，例如：
#   SEARCH=fast_search ./scripts/gen_int8bmodel_mlir.sh [bm1684x|bm1688]
function gen_mlir()
{
    model_transform.py \
        --model_name glc_encoder \
        --model_def ../models/onnx/glc_encoder.onnx \
        --input_shapes "[[1,3,512,768],[1,256,1,1]]" \
        --input_types float32,float32 \
        --mlir glc_encoder.mlir

    model_transform.py \
        --model_name glc_decoder \
        --model_def ../models/onnx/glc_decoder.onnx \
        --input_shapes "[[1,256,32,48],[1,256,8,12],[1,256,1,1]]" \
        --input_types float32,float32,float32 \
        --mlir glc_decoder.mlir
}

function gen_cali_table()
{
    # 编码器校准数据：x 在 [-1,1]，q_enc 循环取 4 个质量档位
    # 由 tools/gen_cali.py 生成到 ../datasets/cali/
    run_calibration.py glc_encoder.mlir \
        --dataset ../datasets/cali/ \
        --input_num ${CALI_NUM:-10} $SEARCH_ARG \
        -o glc_encoder_cali_table

    # 解码器校准数据：先跑编码器 ONNX + CPU 码本查表得到 y / z_hat
    # 由 tools/gen_cali_dec.py 生成到 ../datasets/cali_dec/
    run_calibration.py glc_decoder.mlir \
        --dataset ../datasets/cali_dec/ \
        --input_num ${CALI_NUM:-10} $SEARCH_ARG \
        -o glc_decoder_cali_table
}

function gen_int8bmodel()
{
    model_deploy.py \
        --mlir glc_encoder.mlir \
        --quantize INT8 \
        --chip $target \
        --calibration_table glc_encoder_cali_table \
        --model glc_encoder_int8.bmodel

    model_deploy.py \
        --mlir glc_decoder.mlir \
        --quantize INT8 \
        --chip $target \
        --calibration_table glc_decoder_cali_table \
        --model glc_decoder_int8.bmodel

    mv glc_encoder_int8.bmodel glc_decoder_int8.bmodel $outdir/
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi

SEARCH_ARG=""
[ -n "$SEARCH" ] && SEARCH_ARG="--search $SEARCH"

gen_mlir
gen_cali_table
gen_int8bmodel

popd
