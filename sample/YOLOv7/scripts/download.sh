#!/bin/bash
res=$(which 7z)
if [ $? != 0 ];
then
    echo "Please install 7z on your system!"
    echo "To install, use the following command:"
    echo "sudo apt install p7zip;sudo apt install p7zip-full"
    exit
fi

pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade

scripts_dir=$(cd `dirname $BASH_SOURCE[0]`/ && pwd)

pushd ${scripts_dir}

# datasets
if [ ! -d "../datasets" ]; 
then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv7/datasets.7z
    7z x datasets.7z -o../
    rm datasets.7z
    echo "datasets download!"
else
    echo "datasets exist! Remove it if you need to update."
fi

# models
if [ ! -d "../models" ]; 
then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv7/models.7z
    7z x models.7z -o../
    rm models.7z
    pushd ../models/
    python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv7/BM1688.7z
    7z x BM1688.7z  
    rm -r BM1688.7z
    python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv7/CV186X.7z
    7z x CV186X.7z  
    rm -r CV186X.7z
    popd
    echo "models download!"
else
    echo "models exist! Remove it if you need to update."
fi
popd