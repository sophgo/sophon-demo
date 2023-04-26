#!/bin/bash
pip3 install dfn
# sudo apt install unzip

scripts_dir=$(dirname $(readlink -f "$0"))
# echo $scripts_dir

pushd $scripts_dir
# datasets
if [ ! -d "../datasets" ]; 
then
    python3 -m dfn --url http://disk-sophgo-vip.quickconnect.cn/sharing/4qdi3zIQm

    unzip datasets.zip -d ../
    rm datasets.zip
    echo "datasets download!"
else
    echo "datasets exist!"
fi

# models
if [ ! -d "../models" ]; 
then
    python3 -m dfn --url http://disk-sophgo-vip.quickconnect.cn/sharing/uFNxTRd8B
    unzip models.zip -d ../
    rm models.zip
    echo "models download!"
else
    echo "models exist!"
fi
popd