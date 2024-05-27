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
    mkdir ../datasets
    pushd ../datasets

    python3 -m dfss --url=open@sophgo.com:sophon-demo/common/coco128.tgz
    tar xvf coco128.tgz

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
    pushd onnx
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Real-ESRGAN/models/realesr-general-x4v3.onnx
    popd
    
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Real-ESRGAN/models/BM1688.tgz
    tar xvf BM1688.tgz

    python3 -m dfss --url=open@sophgo.com:sophon-demo/Real-ESRGAN/models/BM1684X.tgz
    tar xvf BM1684X.tgz

    python3 -m dfss --url=open@sophgo.com:sophon-demo/Real-ESRGAN/models/CV186X.tgz
    tar xvf CV186X.tgz
    popd    
    echo "models download!"
else
    echo "Models folder exist! Remove it if you need to update."
fi
popd