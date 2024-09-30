#!/bin/bash
res=$(which unzip)
if [ $? != 0 ];
then
    echo "Please install unzip on your system!"
    exit
fi
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
# sudo apt install unzip

scripts_dir=$(dirname $(readlink -f "$0"))
# echo $scripts_dir

pushd $scripts_dir
# datasets
if [ ! -d "../datasets/" ];
then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/P2PNet/datasets_0524/datasets.zip
    unzip datasets.zip -d ../
    rm datasets.zip

    echo "datasets download!"
else
    echo "Datasets folder exist! Remove it if you need to update."
fi

# models
if [ ! -d "../models" ];
then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/P2PNet/models_0605/models.tar.gz
    tar -xzvf models.tar.gz -C ..
    rm models.tar.gz
    echo "models download!"
else
    echo "Models folder exist! Remove it if you need to update."
fi
popd