#!/bin/bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
scripts_dir=$(dirname $(readlink -f "$0"))

pushd $scripts_dir
# onnx
if [ ! -d "../models" ]; 
then
    mkdir -p ../models
    pushd ../models
    python3 -m dfss --url=open@sophgo.com:sophon-demo/vila/onnx.zip
    unzip onnx.zip
    rm onnx.zip
    popd
    echo "models download!"
else
    echo "Models folder exist! Remove it if you need to update."
fi
popd