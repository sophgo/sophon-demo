#!/bin/bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
scripts_dir=$(dirname $(readlink -f "$0"))

pushd $scripts_dir
# datasets
if [ ! -d "../datasets" ]; 
then
    mkdir ../datasets
    pushd ../datasets
    python3 -m dfss --url=open@sophgo.com:sophon-demo/common/test_obb.tar.gz    #test pictures
    tar xvf test_obb.tar.gz && rm test_obb.tar.gz                               #in case `tar xvf xx` failed.
    python3 -m dfss --url=open@sophgo.com:sophon-demo/common/DOTAv1.tar.gz         #DOTAv1
    tar xvf DOTAv1.tar.gz && rm DOTAv1.tar.gz
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
    python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv8_obb/models/BM1684X.tar.gz
    tar xvf BM1684X.tar.gz && rm BM1684X.tar.gz
    python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv8_obb/models/BM1688.tar.gz
    tar xvf BM1688.tar.gz && rm BM1688.tar.gz
    python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv8_obb/models/CV186X.tar.gz
    tar xvf CV186X.tar.gz && rm CV186X.tar.gz
    python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv8_obb/models/onnx.tar.gz
    tar xvf onnx.tar.gz && rm onnx.tar.gz
    popd
    echo "models download!"
else
    echo "Models folder exist! Remove it if you need to update."
fi

if [ ! -d "../tools/DOTA_devkit_soc" ]; 
then
    pushd ../tools
    python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv8_obb/DOTA_devkit_soc.tar.gz
    tar xvf DOTA_devkit_soc.tar.gz && rm DOTA_devkit_soc.tar.gz
    popd
    echo "DOTA_devkit_soc download!"
else
    echo "Models folder exist! Remove it if you need to update."
fi
popd
