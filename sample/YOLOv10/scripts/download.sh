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
    python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv5/datasets_0918/datasets.zip
    unzip datasets.zip -d ../
    rm datasets.zip

    echo "datasets download!"
else
    echo "Datasets folder exist! Remove it if you need to update."
fi

# models
if [ ! -d "../models" ]; 
then
    mkdir -p ../models
    
    python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv10/tpu-mlir_v1.9.beta.0-116-g3c9d40a6d-20240720.tar.gz
    mv tpu-mlir_v1.9.beta.0-116-g3c9d40a6d-20240720.tar.gz  ../models

    python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv10/BM1684X.zip
    unzip BM1684X.zip -d ../models
    rm BM1684X.zip

    python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv10/BM1688.zip
    unzip BM1688.zip -d ../models
    rm BM1688.zip

    python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv10/CV186X.zip
    unzip CV186X.zip -d ../models
    rm CV186X.zip

    python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv10/onnx.zip
    unzip onnx.zip -d ../models
    rm onnx.zip

    echo "models download!"
else
    echo "Models folder exist! Remove it if you need to update."
fi
popd