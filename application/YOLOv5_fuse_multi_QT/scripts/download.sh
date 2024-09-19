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
    python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv5_fuse/models_240611/BM1688.zip
    unzip BM1688.zip -d ../models
    rm BM1688.zip
    python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv5_fuse/models_240611/CV186X.zip
    unzip CV186X.zip -d ../models
    rm -r CV186X.zip
    echo "models download!"
else
    echo "Models folder exist! Remove it if you need to update."
fi

if [ ! -d "../cpp/install" ];
then
    mkdir -p ../cpp/install
    python3 -m dfss --url=open@sophgo.com:gemini-sdk/qtbase-5.12.8.tar
    tar -xf qtbase-5.12.8.tar --strip-components=1 -C ../cpp/install
    rm -r qtbase-5.12.8.tar
    echo "qtbase download!"
else
    echo "cpp/install folder exist! Remove it if you need to update."
fi
popd