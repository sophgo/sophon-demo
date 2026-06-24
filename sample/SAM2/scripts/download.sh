#!/bin/bash
res=$(which unzip)
if [ $? != 0 ];
then
    echo "Please install unzip on your system!"
    exit
fi
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
scripts_dir=$(dirname $(readlink -f "$0"))

pushd $scripts_dir
# datasets
if [ ! -d "../datasets" ]; 
then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/SAM2/datasets.zip
    unzip datasets.zip -d ../datasets
    rm datasets.zip

    pushd ../datasets
    python3 -m dfss --url=open@sophgo.com:sophon-demo/SAM2/cali_npz.tar.gz
    tar xvf cali_npz.tar.gz
    rm cali_npz.tar.gz
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
    python3 -m dfss --url=open@sophgo.com:sophon-demo/SAM2/BM1684X.tar.gz
    tar xvf BM1684X.tar.gz
    rm BM1684X.tar.gz

    python3 -m dfss --url=open@sophgo.com:sophon-demo/SAM2/BM1688.tar.gz
    tar xvf BM1688.tar.gz
    rm BM1688.tar.gz
    popd

    python3 -m dfss --url=open@sophgo.com:sophon-demo/SAM2/onnx.zip
    unzip onnx.zip -d ../models
    rm onnx.zip

    python3 -m dfss --url=open@sophgo.com:sophon-demo/SAM2/torch.zip
    unzip torch.zip -d ../models
    rm torch.zip

    echo "models download!"
else
    echo "Models folder exist! Remove it if you need to update."
fi
popd