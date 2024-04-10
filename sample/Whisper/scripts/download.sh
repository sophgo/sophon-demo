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
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Whisper/datasets_240327/datasets.zip
    unzip datasets.zip -d ../
    rm datasets.zip

    echo "datasets download!"
else
    echo "Datasets folder exist! Remove it if you need to update."
fi

# models
if [ ! -d "../models" ];
then
    mkdir ../models
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Whisper/model_240408/bmodel.zip
    unzip bmodel.zip -d ../models
    rm bmodel.zip
    echo "bmodel download!"
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Whisper/model_240408/onnx.zip
    unzip onnx.zip -d ../models
    rm onnx.zip
    echo "onnx models download!"
else
    echo "Models folder exist! Remove it if you need to update."
fi
# assets
if [ ! -d "../python/bmwhisper/assets" ];
then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Whisper/model_240408/assets.zip
    unzip assets.zip -d ../python/bmwhisper
    rm assets.zip
    echo "assets download!"
else
    echo "Assets folder exist! Remove it if you need to update."
fi
popd