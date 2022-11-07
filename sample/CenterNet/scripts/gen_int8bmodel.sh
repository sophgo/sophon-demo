 #!/bin/bash

source model_info.sh

SCRIPT_DIR=`pwd`/`dirname $0`
MODEL_DIR=$SCRIPT_DIR/../data/models
model_target=${1:-BM1684}

if [ ! -d "$MODEL_DIR/$model_target" ]; then
  echo "create $model_target dir: $MODEL_DIR/$model_target"
  mkdir -p $MODEL_DIR/$model_target
fi

python3 -m ufw.cali.cali_model \
             --net_name 'centernet' \
             --model ${MODEL_DIR}/torch/ctdet_coco_dlav0_1x.torchscript.pt \
             --cali_image_path ../data/images/ \
             --cali_image_preprocess='resize_h=512,resize_w=512;mean_value=104.01195:114.03422:119.91659, scale=0.014' \
             --input_shapes [4,3,512,512] \
             --fp32_layer_list '30,33,36' \
             --target $model_target
            

cp ../data/models/torch/centernet_batch4/compilation.bmodel $MODEL_DIR/$model_target/centernet_int8_4b.bmodel
echo "[Success] $MODEL_DIR/$model_target/centernet_int8_4b.bmodel generated."

python3 -m ufw.cali.cali_model \
             --net_name 'centernet' \
             --model ${MODEL_DIR}/torch/ctdet_coco_dlav0_1x.torchscript.pt \
             --cali_image_path ../data/images/ \
             --cali_image_preprocess='resize_h=512,resize_w=512;mean_value=104.01195:114.03422:119.91659, scale=0.014' \
             --input_shapes [1,3,512,512] \
             --fp32_layer_list '30,33,36' \
             --target $model_target
            

cp ../data/models/torch/centernet_batch1/compilation.bmodel $MODEL_DIR/$model_target/centernet_int8_1b.bmodel
echo "[Success] $MODEL_DIR/$model_target/centernet_int8_1b.bmodel generated."