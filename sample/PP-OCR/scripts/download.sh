#!/bin/bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
scripts_dir=$(dirname $(readlink -f "$0"))

pushd $scripts_dir
# datasets
if [ ! -d "../datasets" ]; 
then
    pushd ../
    python3 -m dfss --url=open@sophgo.com:sophon-demo/PP-OCR/datasets.tar.gz
    tar xvf datasets.tar.gz && rm datasets.tar.gz
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
    python3 -m dfss --url=open@sophgo.com:sophon-demo/PP-OCR/models_v4/BM1684.tar.gz
    tar xvf BM1684.tar.gz && rm BM1684.tar.gz
    python3 -m dfss --url=open@sophgo.com:sophon-demo/PP-OCR/models_v4/BM1684X.tar.gz
    tar xvf BM1684X.tar.gz && rm BM1684X.tar.gz
    python3 -m dfss --url=open@sophgo.com:sophon-demo/PP-OCR/models_v4/BM1688.tar.gz
    tar xvf BM1688.tar.gz && rm BM1688.tar.gz
    python3 -m dfss --url=open@sophgo.com:sophon-demo/PP-OCR/models_v4/CV186X.tar.gz
    tar xvf CV186X.tar.gz && rm CV186X.tar.gz
    python3 -m dfss --url=open@sophgo.com:sophon-demo/PP-OCR/models_v4/onnx.tar.gz
    tar xvf onnx.tar.gz && rm onnx.tar.gz
    python3 -m dfss --url=open@sophgo.com:sophon-demo/PP-OCR/models_v4/paddle.tar.gz
    tar xvf paddle.tar.gz && rm paddle.tar.gz
    popd
    echo "models download!"
else
    echo "Models folder exist! Remove it if you need to update."
fi
popd