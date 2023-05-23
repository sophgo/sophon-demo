#!/bin/bash
pip3 install dfn
# sudo apt install unzip

scripts_dir=$(dirname $(readlink -f "$0"))
# echo $scripts_dir

pushd $scripts_dir
# data
if [ ! -d "../datasets/" ]; 
then
    # ShanghaiTech.zip
    python3 -m dfn --url http://disk-sophgo-vip.quickconnect.cn/sharing/TV0e5stAd
    unzip ShanghaiTech.zip -d ../datasets/
    rm ShanghaiTech.zip

    # video.zip
    python3 -m dfn --url http://disk-sophgo-vip.quickconnect.cn/sharing/0EFl4Djfz
    unzip video.zip -d ../datasets/
    rm video.zip

    echo "datasets download!"
else
    echo "datasets exist!"
fi

# models
if [ ! -d "../models" ]; 
then
    python3 -m dfn --url http://disk-sophgo-vip.quickconnect.cn/sharing/aZcHkJNiG
    unzip models.zip -d ../models/
    rm models.zip
    echo "models download!"
else
    echo "models exist!"
fi
popd