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
    python3 -m dfss --url=open@sophgo.com:sophon-demo/SAM2/datasets.zip
    unzip datasets.zip -d ../datasets
    rm datasets.zip

    echo "datasets download!"
else
    echo "Datasets folder exist! Remove it if you need to update."
fi

# models
if [ ! -d "../models" ]; 
then
    mkdir ../models

    python3 -m dfss --url=open@sophgo.com:sophon-demo/SAM2/BM1688.zip
    unzip BM1688.zip -d ../models
    rm BM1688.zip

    python3 -m dfss --url=open@sophgo.com:sophon-demo/SAM2/onnx.zip
    unzip onnx.zip -d ../models
    rm onnx.zip

    python3 -m dfss --url=open@sophgo.com:sophon-demo/SAM2/torch.zip
    unzip torch.zip -d ../models
    rm torch.zip

    echo "models download!"
else
    echo "Models folder exist! Remove it if you need to update."
fi
popd