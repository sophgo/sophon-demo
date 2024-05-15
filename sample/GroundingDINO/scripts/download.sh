#!/bin/bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade

res=$(which unzip)
if [ $? != 0 ];
then
    echo "Please install unzip on your system!"
    exit
fi

echo "You has already installed unzip!"

scripts_dir=$(dirname $(readlink -f "$0"))
# echo $scripts_dir

pushd $scripts_dir
# datasets
if [ ! -d "../datasets" ]; 
then
    mkdir -p ../datasets

    python3 -m dfss --url=open@sophgo.com:sophon-demo/GroundingDINO/datasets.zip
    unzip datasets.zip -d ../datasets
    rm datasets.zip

    echo "datasets download!"
else
    echo "datasets exist!"
fi

# models
if [ ! -d "../models" ]; 
then
    mkdir ../models
    python3 -m dfss --url=open@sophgo.com:sophon-demo/GroundingDINO/models.zip
    unzip models.zip -d ../models
    rm models.zip

    python3 -m dfss --url=open@sophgo.com:sophon-demo/GroundingDINO/GroundingDINO.tar
    mv GroundingDINO.tar ../models

    python3 -m dfss --url=open@sophgo.com:sophon-demo/GroundingDINO/bert-base-uncased.zip
    unzip bert-base-uncased.zip -d ../models
    rm bert-base-uncased.zip

    python3 -m dfss --url=open@sophgo.com:sophon-demo/GroundingDINO/onnx.zip
    unzip onnx.zip -d ../models
    rm onnx.zip

    echo "models download!"
else
    echo "models exist!"
fi
