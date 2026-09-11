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
    python3 -m dfss --url=open@sophgo.com:sophon-demo/FaceFormer/datasets.zip
    unzip datasets.zip -d ../
    rm datasets.zip
    echo "datasets download!"
else
    echo "datasets folder exist! Remove it if you need to update."
fi

# models
if [ ! -d "../models" ];
then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/FaceFormer/models.zip
    unzip models.zip -d ../
    rm models.zip
    echo "models download!"
else
    echo "Models folder exist! Remove it if you need to update."
fi

# BM1684X2 models
if [ ! -d "../models/BM1684X2" ];
then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/FaceFormer/models_260906/BM1684X2.zip
    unzip BM1684X2.zip -d ../models/
    rm BM1684X2.zip
    echo "BM1684X2 models download!"
else
    echo "BM1684X2 models folder exist! Remove it if you need to update."
fi

if [ ! -d "../tools/wav2vec2-base-960h" ] || [ ! -d "../tools/vocaset" ];
then
    pushd ../tools
    python3 -m dfss --url=open@sophgo.com:sophon-demo/FaceFormer/tools_dev.zip
    unzip tools_dev.zip
    rm tools_dev.zip
    popd
else
    echo "wav2vec2-base-960h or vocaset exists! Remove it if you need to update."
fi

popd