#!/bin/bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
scripts_dir=$(dirname $(readlink -f "$0"))

pushd $scripts_dir
# datasets
if [ ! -d "../datasets" ]; 
then
    mkdir ../datasets
    pushd ../datasets
    python3 -m dfss --url=open@sophgo.com:sophon-demo/VITS_CHINESE/vits_infer_item.txt
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
    python3 -m dfss --url=open@sophgo.com:sophon-demo/VITS_CHINESE/models/bmodel/BM1684X.tar.gz
    python3 -m dfss --url=open@sophgo.com:sophon-demo/VITS_CHINESE/models/bmodel/BM1688.tar.gz
    python3 -m dfss --url=open@sophgo.com:sophon-demo/VITS_CHINESE/models/bmodel/CV186X.tar.gz
    tar xvf BM1684X.tar.gz && rm BM1684X.tar.gz
    tar xvf BM1688.tar.gz && rm BM1688.tar.gz
    tar xvf CV186X.tar.gz && rm CV186X.tar.gz
    popd
    echo "models download!"
else
    echo "Models folder exist! Remove it if you need to update."
fi
popd
