#!/bin/bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
scripts_dir=$(dirname $(readlink -f "$0"))

pushd $scripts_dir
# onnx
if [ ! -d "../models" ]; 
then
    mkdir -p ../models
fi

pushd ../models
python3 -m dfss --url=open@sophgo.com:sophon-demo/vila/export_onnx.zip
unzip export_onnx.zip
rm export_onnx.zip
popd
echo "models download!"
popd