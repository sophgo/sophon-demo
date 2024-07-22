#!/bin/bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
scripts_dir=$(dirname $(readlink -f "$0"))

pushd $scripts_dir
# datasets
if [ ! -d "../datasets" ]; 
then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/SuperGlue/datasets.tar.gz
    tar xvf datasets.tar.gz -C ../ && rm datasets.tar.gz
    echo "datasets download!"
else
    echo "Datasets folder exist! Remove it if you need to update."
fi

# models
if [ ! -d "../models" ]; 
then
    mkdir ../models
    pushd ../models
    python3 -m dfss --url=open@sophgo.com:sophon-demo/SuperGlue/models/BM1684X.tar.gz
    tar xvf BM1684X.tar.gz && rm BM1684X.tar.gz
    python3 -m dfss --url=open@sophgo.com:sophon-demo/SuperGlue/models/BM1688.tar.gz
    tar xvf BM1688.tar.gz && rm BM1688.tar.gz
    python3 -m dfss --url=open@sophgo.com:sophon-demo/SuperGlue/models/CV186X.tar.gz
    tar xvf CV186X.tar.gz && rm CV186X.tar.gz
    python3 -m dfss --url=open@sophgo.com:sophon-demo/SuperGlue/models/onnx.tar.gz
    tar xvf onnx.tar.gz && rm onnx.tar.gz
    python3 -m dfss --url=open@sophgo.com:sophon-demo/SuperGlue/models/weights.tar.gz
    tar xvf weights.tar.gz && rm weights.tar.gz
    echo "models download!"
    popd
else
    echo "Models folder exist! Remove it if you need to update."
fi

# cpp_dependencies
if [ ! -d "../cpp/libtorch" ]; 
then
    pushd ../cpp
    python3 -m dfss --url=open@sophgo.com:sophon-demo/SuperGlue/libtorch.tar.gz
    tar xvf libtorch.tar.gz && rm libtorch.tar.gz
    popd
    echo "x86 libtorch download!"
else
    echo "libtorch folder exist! Remove it if you need to update."
fi

if [ ! -d "../cpp/aarch64_lib" ]; 
then
    pushd ../cpp
    python3 -m dfss --url=open@sophgo.com:sophon-demo/SuperGlue/aarch64_lib.tar.gz
    tar xvf aarch64_lib.tar.gz && rm aarch64_lib.tar.gz
    popd
    echo "aarch64_lib download!"
else
    echo "aarch64_lib folder exist! Remove it if you need to update."
fi
popd