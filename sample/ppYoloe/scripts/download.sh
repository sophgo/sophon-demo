#!/bin/bash
pip3 install dfss

res=$(which unzip)
if [ $? != 0 ];
then
    echo "Please install unzip on your system!"
    echo "To install, use the following command:"
    echo "sudo apt install unzip"
    exit
fi

scripts_dir=$(dirname $(readlink -f "$0"))
# echo $scripts_dir

pushd $scripts_dir

# datasets
if [ ! -d "../datasets" ];
then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv5/datasets_0918/datasets.zip
    unzip datasets.zip -d ../
    rm datasets.zip

    echo "datasets download!"
else
    echo "datasets exist!"
fi

# models
if [ ! -d "../models" ];
then
    mkdir -p ../models
    python3 -m dfss --url=open@sophgo.com:sophon-demo/ppYoloe/models.zip
    unzip models.zip -d ../models
    rm models.zip
    echo "models download!"
else
    echo "models exist!"
fi
popd
