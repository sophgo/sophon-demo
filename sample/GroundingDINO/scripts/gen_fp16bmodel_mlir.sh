#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))

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

gen_mlir()
{
    model_transform.py \
        --model_name groundingdino \
        --model_def ../models/onnx/GroundingDino.onnx \
        --input_shapes [[1,3,800,800],[1,256],[1,256,256],[1,256],[1,256],[1,256,256],[1,256],[1,13294,4]]  \
        --mlir groundingdino.mlir
}

gen_fp16bmodel()
{
    model_deploy.py \
        --mlir groundingdino.mlir \
        --quantize F16 \
        --chip ${target} \
        --model groundingdino_${target}_fp16.bmodel

    mv groundingdino_${target}_fp16.bmodel $outdir/
}

# BM1684X2: tpu.TopK(use_hau=true) 会被 SubnetDivide 强制切进 TPU_DYNAMIC(HAU) 子网，
# 当前固件跑该路径会挂死 TPU（bmrt_test 卡住、driver wait api timeout）。
# 这里在 stage-1 产物上把 use_hau 改成 false（走静态多核路径），再手动跑 stage-2 + codegen。
gen_fp16bmodel_bm1684x2()
{
    model_deploy.py \
        --mlir groundingdino.mlir \
        --quantize F16 \
        --chip bm1684x2 \
        --tolerance 0.95 0.89 \
        --model groundingdino_bm1684x2_f16.bmodel \
        --asymmetric_data 0 2>&1 | tail -n 4

    # 将 TopK 的 use_hau 置 false（约在 mlir 第 5605 行，loc "/transformer/TopK"）
    sed -i 's/\(tpu\.TopK.*\)use_hau = true/\1use_hau = false/' groundingdino_bm1684x2_f16_tpu.mlir

    tpuc-opt \
        groundingdino_bm1684x2_f16_tpu.mlir \
        --mlir-disable-threading \
        --strip-io-quant="quant_input=False quant_output=False quant_input_list= quant_output_list= quant_output_bf16=False quant_output_int8=False" \
        --processor-tpu-optimize \
        --dev-parallel \
        --weight-reorder \
        --subnet-divide="dynamic=False" \
        --op-reorder \
        --topo-sort \
        --layer-group="opt=2 group_by_cores=auto compress_mode=none debugger=0 disable_group_overlap=false lgcache=true config_filename= enable_lghash=False lghash_dir=" \
        --affine-opt \
        --core-parallel \
        --after-layergroup-weight-reorder \
        --address-assign \
        -o groundingdino_bm1684x2_f16_final.mlir

    tpuc-opt \
        groundingdino_bm1684x2_f16_final.mlir \
        --mlir-disable-threading \
        --codegen="model_file=groundingdino_bm1684x2_fp16.bmodel embed_debug_info=False model_version=latest bmodel_only=False gdma_check=False rvti=False" \
        -o /dev/null

    mv groundingdino_bm1684x2_fp16.bmodel $outdir/
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi

# batch size 1
if test $target = "bm1684x2"; then
    gen_mlir 1
    gen_fp16bmodel_bm1684x2 1
else
    gen_mlir 1
    gen_fp16bmodel 1
fi

popd
