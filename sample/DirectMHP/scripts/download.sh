#!/bin/bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
scripts_dir=$(dirname $(readlink -f "$0"))

pushd $scripts_dir
# datasets
if [ ! -d "../datasets" ]; 
then
    mkdir ../datasets
    pushd ../datasets
    python3 -m dfss --url=open@sophgo.com:sophon-demo/DirectMHP/datasets/test.tar.gz    #test pictures
    tar xvf test.tar.gz && rm test.tar.gz                                   #in case `tar xvf xx` failed.
    python3 -m dfss --url=open@sophgo.com:sophon-demo/DirectMHP/datasets/val_pictures.tar.gz
    tar xvf val_pictures.tar.gz && rm val_pictures.tar.gz
    python3 -m dfss --url=open@sophgo.com:sophon-demo/DirectMHP/datasets/person_small.mp4 #test video
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
    python3 -m dfss --url=open@sophgo.com:sophon-demo/DirectMHP/models/BM1684X.tar.gz
    tar xvf BM1684X.tar.gz && rm BM1684X.tar.gz
    python3 -m dfss --url=open@sophgo.com:sophon-demo/DirectMHP/models/BM1688.tar.gz
    tar xvf BM1688.tar.gz && rm BM1688.tar.gz
    python3 -m dfss --url=open@sophgo.com:sophon-demo/DirectMHP/models/CV186X.tar.gz
    tar xvf CV186X.tar.gz && rm CV186X.tar.gz
    python3 -m dfss --url=open@sophgo.com:sophon-demo/DirectMHP/models/onnx.tar.gz
    tar xvf onnx.tar.gz && rm onnx.tar.gz
    python3 -m dfss --url=open@sophgo.com:sophon-demo/DirectMHP/models/torch.tar.gz
    tar xvf torch.tar.gz && rm torch.tar.gz
    popd
    echo "models download!"
else
    echo "Models folder exist! Remove it if you need to update."
fi
popd