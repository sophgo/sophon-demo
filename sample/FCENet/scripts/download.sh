#!/bin/bash
pip3 install dfn

scripts_dir=$(dirname $(readlink -f "$0"))
# echo $scripts_dir

pushd $scripts_dir

mkdir -p ../data/images

# test dataset
python3 -m dfn --url http://disk-sophgo-vip.quickconnect.cn/sharing/MnCq5ZOyn
tar -xf ctw1500.tar -C ../datasets/
rm ctw1500.tar

# models
python3 -m dfn --url http://disk-sophgo-vip.quickconnect.cn/sharing/GwqUqVEGS
tar -xf models.tar -C ../
rm models.tar

popd
