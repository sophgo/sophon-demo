#!/bin/bash

source model_info.sh

SCRIPT_DIR=`pwd`/`dirname $0`
MODEL_DIR=$SCRIPT_DIR/../data/models/
model_target=${1:-BM1684}
fp32model_dir="$build_dir/fp32model_${model_target}"

if [ ! -d "$MODEL_DIR/$model_target" ]; then
  echo "create $model_target dir: $MODEL_DIR/$model_target"
  mkdir -p $MODEL_DIR/$model_target
fi

python3 \
    -m bmnetp \
    --net_name=centernet \
    --target=$model_target \
    --opt=2 \
    --cmp=true \
    --enable_profile=true \
    --shapes=[1,3,512,512] \
    --model=$MODEL_DIR/torch/ctdet_coco_dlav0_1x.torchscript.pt \
    --outdir=$fp32model_dir \
    --dyn=false

cp $fp32model_dir/compilation.bmodel $MODEL_DIR/$model_target/centernet_fp32_1b.bmodel

echo "[Success] $MODEL_DIR/$model_target/centernet_fp32_1b.bmodel generated."

python3 \
    -m bmnetp \
    --net_name=centernet \
    --target=$model_target \
    --opt=2 \
    --cmp=true \
    --enable_profile=true \
    --shapes=[4,3,512,512] \
    --model=$MODEL_DIR/torch/ctdet_coco_dlav0_1x.torchscript.pt \
    --outdir=$fp32model_dir \
    --dyn=false

cp $fp32model_dir/compilation.bmodel $MODEL_DIR/$model_target/centernet_fp32_4b.bmodel

echo "[Success] $MODEL_DIR/$model_target/centernet_fp32_4b.bmodel generated."