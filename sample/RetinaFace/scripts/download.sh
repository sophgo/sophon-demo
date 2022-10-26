#!/bin/bash
pip3 install dfn

scripts_dir=$(dirname $(readlink -f "$0"))
# echo $scripts_dir

pushd $scripts_dir

mkdir -p ../data/images
mkdir -p ../data/videos
mkdir -p ../data/images/face

# 测试集WIDER FACE
python3 -m dfn --url http://219.142.246.77:65000/sharing/4ryx2MOgm
tar -xf WIDERVAL.tar -C ../data/images/
rm WIDERVAL.tar

## 单张图片与视频
python3 -m dfn --url http://219.142.246.77:65000/sharing/BaQm9vdpp
tar -xf test.tar -C ../data
mv ../data/images/face0*.jpg ../data/images/face
rm test.tar

# models
# 包含原始pytorch模型、traced pytorch模型、fp32bmodel
python3 -m dfn --url http://219.142.246.77:65000/sharing/7AyD917fx
tar -xf models.tar -C ../data/
rm models.tar


popd
