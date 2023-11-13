#!/bin/bash
pip3 install dfn
sudo apt install unzip

script_dir=$(dirname $(readlink -f "$0"))
echo $script_dir

pushd $script_dir

# datasets
if [ ! -d "../datasets" ];
then
    mkdir -p ../datasets
    # test dataset
    # test input_test
    python3 -m dfss --url=open@sophgo.com:sophon-demo/segformer/datasets.zip
    unzip datasets.zip -d ../
    rm datasets.zip
    echo "datasets download!"
else
    echo "datasets exist!"
fi


# models
if [ ! -d "../models" ]; 
then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/segformer/models.zip
    ls -al
    unzip models.zip -d ../
    rm models.zip
    echo "models download!"
else
    echo "models exist!"
fi

popd
