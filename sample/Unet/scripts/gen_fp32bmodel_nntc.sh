#!/bin/bash

SCRIPT_DIR=`pwd`/`dirname $0`
MODEL_DIR=$SCRIPT_DIR/../models
model_target=BM1684

python3 \
    -m bmnetp \
    --net_name=unet \
    --target=$model_target \
    --opt=2 \
    --cmp=true \
    --enable_profile=true \
    --shapes=[1,3,640,959] \
    --model=$MODEL_DIR/torch/unet.pt \
    --outdir=./bmodel_1684 \
    --dyn=false

cp ./bmodel_1684/compilation.bmodel $MODEL_DIR/BM1684/unet_fp32_1b.bmodel

echo "[Success] $MODEL_DIR/BM1684/unet_fp32_1b.bmodel generated."


model_target=BM1684X

python3 \
    -m bmnetp \
    --net_name=unet \
    --target=$model_target \
    --opt=2 \
    --cmp=true \
    --enable_profile=true \
    --shapes=[1,3,640,959] \
    --model=$MODEL_DIR/torch/unet.pt \
    --outdir=./bmodel_1684X \
    --dyn=false

cp ./bmodel_1684X/compilation.bmodel $MODEL_DIR/BM1684X/unet_fp32_1b.bmodel

echo "[Success] $MODEL_DIR/BM1684X/unet_fp32_1b.bmodel generated."