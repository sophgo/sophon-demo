#!/bin/bash
res=$(dpkg -l|grep unzip)
if [ $? != 0 ];
then
    echo "Please install unzip on your system!"
    exit
fi
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
scripts_dir=$(dirname $(readlink -f "$0"))

pushd $scripts_dir
if [ ! -d "../data" ]; 
then
    mkdir -p ../data/
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Retinaface/data_0918/images.zip
    unzip images.zip -d ../data/
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Retinaface/data_0918/videos.zip
    unzip videos.zip -d ../data/
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Retinaface/data_0918/models.zip
    unzip models.zip -d ../data/S
    rm images.zip videos.zip models.zip
else
    echo "Data folder exist! Remove it if you need to update."
fi

popd
