#!/bin/bash
pip3 install dfn

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

    python3 -m dfss --url=open@sophgo.com:sophon-demo/yolact/datasets.zip
    unzip datasets.zip -d ../datasets
    rm datasets.zip

    echo "datasets download!"
else
    echo "datasets exist!"
fi

# models
if [ ! -d "../models" ]; 
then
    mkdir -p ../models

    python3 -m dfss --url=open@sophgo.com:sophon-demo/yolact/models.zip
    unzip models.zip -d ../models
    rm models.zip

    echo "models download!"
else
    echo "models exist!"
fi
popd
