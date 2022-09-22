#!/bin/bash
pip3 install dfn

scripts_dir=$(dirname $(readlink -f "$0"))
# echo $scripts_dir

pushd $scripts_dir

mkdir -p ../data/images
mkdir -p ../data/videos
mkdir -p ../data/images/face

## 单张图片与视频
python3 -m dfn --url http://219.142.246.77:65000/sharing/YYIgBtODP
tar -xf test.tar -C ../data
mv ../data/images/face0*.jpg ../data/images/face
rm test.tar

# models
# 包含原始pytorch模型、traced pytorch模型、fp32bmodel和int8bmodel
python3 -m dfn --url http://219.142.246.77:65000/sharing/7xcffMHoJ
tar -xf models.tar -C ../data/
rm models.tar

# cali dataset
python3 -m dfn --url http://219.142.246.77:65000/sharing/3Uk81fcFr
tar -xf cali_dataset.tar -C ../data/images/
rm cali_dataset.tar

popd
