#!/bin/bash
pip3 install dfn

scripts_dir=$(dirname $(readlink -f "$0"))
# echo $scripts_dir

pushd $scripts_dir

mkdir -p ../data/images

# test dataset
# imagenet验证集随机抽取1000张
python3 -m dfn --url http://219.142.246.77:65000/sharing/kKHeokzEF
tar -xf imagenet_val_1k.tar -C ../data/images/
rm imagenet_val_1k.tar

# models
# 包含原始pytorch模型、traced pytorch模型、fp32bmodel和int8bmodel
python3 -m dfn --url http://219.142.246.77:65000/sharing/kcI1V2G2f
tar -xf models.tar -C ../data/
rm models.tar

# cali data
# imagenet验证集随机抽出了200张
python3 -m dfn --url http://219.142.246.77:65000/sharing/gjR0Cpw2u
tar -xf cali_data.tar -C ../data/images/
rm cali_data.tar

popd
