#!/bin/bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
scripts_dir=$(dirname $(readlink -f "$0"))

pushd $scripts_dir
# datasets
if [ ! -d "../datasets" ]; 
then
    mkdir ../datasets
    pushd ../datasets
    python3 -m dfss --url=open@sophgo.com:sophon-demo/MP_SENet/datasets/test.wav
    popd
    echo "datasets download!"
else
    echo "Datasets folder exist! Remove it if you need to update."
fi


# models
if [ ! -d "../models" ]; 
then
    mkdir ../models
    pushd ../models
    mkdir onnx
    mkdir torch
    python3 -m dfss --url=open@sophgo.com:sophon-demo/MP_SENet/models/bmodel/BM1684X.tar.gz
    
    tar xvf BM1684X.tar.gz && rm BM1684X.tar.gz

    popd
    echo "models download!"
else
    echo "Models folder exist! Remove it if you need to update."
fi
popd
