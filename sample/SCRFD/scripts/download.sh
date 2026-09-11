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
    python3 -m dfss --url=open@sophgo.com:sophon-demo/SCRFD/datasets.zip
    unzip datasets.zip -d ../
    rm datasets.zip

    echo "datasets download!"
else
    echo "datasets exist!"
fi

# models
if [ ! -d "../models" ];
then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/SCRFD/models.zip
    unzip models.zip -d ../
    rm models.zip
    echo "models download!"
else
    echo "models exist!"
fi

# BM1684X2 bmodels (单独提供，models.zip 中未包含)
if [ ! -d "../models/BM1684X2" ];
then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/SCRFD/BM1684X2.tar.gz
    tar xvf BM1684X2.tar.gz -C ../models/ && rm BM1684X2.tar.gz
    echo "models/BM1684X2 download!"
else
    echo "models/BM1684X2 exist!"
fi

# ground_truth
if [ ! -d "../tools/ground_truth" ];
then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/SCRFD/ground_truth.zip
    unzip ground_truth.zip -d ../tools/
    rm ground_truth.zip
    echo "ground_truth download!"
else
    echo "ground_truth exist!"
fi
popd
