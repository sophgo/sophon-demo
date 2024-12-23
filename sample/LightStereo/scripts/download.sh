#!/bin/bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
scripts_dir=$(dirname $(readlink -f "$0"))

pushd $scripts_dir
# datasets
if [ ! -d "../datasets" ]; 
then
    mkdir ../datasets
    pushd ../datasets
    python3 -m dfss --url=open@sophgo.com:sophon-demo/LightStereo/cali_data.tar.gz
    tar xvf cali_data.tar.gz && rm cali_data.tar.gz
    python3 -m dfss --url=open@sophgo.com:sophon-demo/LightStereo/KITTI12.tar.gz
    tar xvf KITTI12.tar.gz && rm KITTI12.tar.gz
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
    python3 -m dfss --url=open@sophgo.com:sophon-demo/LightStereo/models/BM1684X.tar.gz
    tar xvf BM1684X.tar.gz && rm BM1684X.tar.gz
    python3 -m dfss --url=open@sophgo.com:sophon-demo/LightStereo/models/BM1688.tar.gz
    tar xvf BM1688.tar.gz && rm BM1688.tar.gz
    python3 -m dfss --url=open@sophgo.com:sophon-demo/LightStereo/models/CV186X.tar.gz
    tar xvf CV186X.tar.gz && rm CV186X.tar.gz
    python3 -m dfss --url=open@sophgo.com:sophon-demo/LightStereo/models/ckpt.tar.gz
    tar xvf ckpt.tar.gz && rm ckpt.tar.gz
    python3 -m dfss --url=open@sophgo.com:sophon-demo/LightStereo/models/onnx.tar.gz
    tar xvf onnx.tar.gz && rm onnx.tar.gz
    popd
    echo "models download!"
else
    echo "Models folder exist! Remove it if you need to update."
fi
popd