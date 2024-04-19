#!/bin/bash
res=$(which unzip)
if [ $? != 0 ];
then
    echo "Please install unzip on your system!"
    echo "To install, use the following command:"
    echo "sudo apt install unzip"
    exit
fi
res=$(which tar)
if [ $? != 0 ];
then
    echo "Please install tar on your system!"
    echo "To install, use the following command:"
    echo "sudo apt install tar"
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
    mkdir -p ../models/
    pushd ../models/
    python3 -m dfss --url=open@sophgo.com:sophon-demo/ppYOLOv3/models_20240416/BM1684.tgz
    tar xvf BM1684.tgz
    python3 -m dfss --url=open@sophgo.com:sophon-demo/ppYOLOv3/models_20240416/BM1684X.tgz
    tar xvf BM1684X.tgz
    python3 -m dfss --url=open@sophgo.com:sophon-demo/ppYOLOv3/models_20240416/BM1688.tgz
    tar xvf BM1688.tgz
    python3 -m dfss --url=open@sophgo.com:sophon-demo/ppYOLOv3/models_20240416/CV186X.tgz
    tar xvf CV186X.tgz
    rm -r *.tgz
    mkdir onnx
    pushd onnx
    python3 -m dfss --url=open@sophgo.com:sophon-demo/ppYOLOv3/models_20240416/onnx/ppyolov3_1b.onnx
    popd
    popd
    echo "models download!"
else
    echo "Models folder exist! Remove it if you need to update."
fi
popd