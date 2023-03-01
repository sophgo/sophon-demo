#!/bin/bash
pip3 install dfn
# sudo apt install unzip

scripts_dir=$(dirname $(readlink -f "$0"))
# echo $scripts_dir

pushd $scripts_dir

# datasets
if [ ! -d "../datasets" ]; 
then
    mkdir -p ../datasets
    # test dataset
    # imagenet验证集随机抽取1000张
    python3 -m dfn --url http://219.142.246.77:65000/sharing/oh2B5pGUu
    unzip imagenet_val_1k.zip -d ../datasets/
    rm imagenet_val_1k.zip

    # cali data
    # imagenet验证集随机抽出了200张
    python3 -m dfn --url http://219.142.246.77:65000/sharing/W1ED8v06A
    unzip cali_data.zip -d ../datasets/
    rm cali_data.zip

    echo "datasets download!"
else
    echo "datasets exist!"
fi

# models
if [ ! -d "../models" ]; 
then
    # 包含原始pytorch模型、traced pytorch模型、fp32bmodel、fp16bmodel和int8bmodel
    python3 -m dfn --url http://219.142.246.77:65000/sharing/dHWRblq7M
    unzip models.zip -d ../
    rm models.zip
    echo "models download!"
else
    echo "models exist!"
fi
popd
