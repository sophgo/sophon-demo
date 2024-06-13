#!/bin/bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
scripts_dir=$(dirname $(readlink -f "$0"))

pushd $scripts_dir
# datasets
if [ ! -d "../datasets" ]; 
then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/SuperGlue/datasets.tar.gz
    tar xvf datasets.tar.gz -C ../
    echo "datasets download!"
else
    echo "Datasets folder exist! Remove it if you need to update."
fi

# models
if [ ! -d "../models" ]; 
then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/SuperGlue/models.tar.gz
    tar xvf models.tar.gz -C ../
    popd
    echo "models download!"
else
    echo "Models folder exist! Remove it if you need to update."
fi

# cpp_dependencies
if [ ! -d "../cpp/libtorch" ]; 
then
    mkdir ../cpp
    pushd ../cpp
    python3 -m dfss --url=open@sophgo.com:sophon-demo/SuperGlue/libtorch.tar.gz
    tar xvf libtorch.tar.gz
    popd
    echo "x86 libtorch download!"
else
    echo "libtorch folder exist! Remove it if you need to update."
fi

if [ ! -d "../cpp/aarch64_lib" ]; 
then
    mkdir ../cpp
    pushd ../cpp
    python3 -m dfss --url=open@sophgo.com:sophon-demo/SuperGlue/aarch64_lib.tar.gz
    tar xvf aarch64_lib.tar.gz
    popd
    echo "aarch64_lib download!"
else
    echo "aarch64_lib folder exist! Remove it if you need to update."
fi
popd