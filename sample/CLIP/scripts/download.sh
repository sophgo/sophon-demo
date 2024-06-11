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

# models
if [ ! -d "../models" ];
then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/CLIP/models.zip
    unzip models.zip -d ../
    rm models.zip
    popd
    echo "models download!"
else
    echo "Models folder exist! Remove it if you need to update."
fi
popd