#!/bin/bash
pip3 install dfn
sudo apt install unzip

script_dir=$(dirname $(readlink -f "$0"))
echo $script_dir

pushd $script_dir

# datasets
if [ ! -d "../datasets" ];
then
    mkdir -p ../datasets
    # test dataset
    # test input_test
    python3 -m dfn --url http://disk-sophgo-vip.quickconnect.cn/sharing/8QitHtqqJ
    mv carvana_video.mp4 ../datasets/carvana_video.mp4
    echo "[Success] carvana_video.mp4 has been downloaded to path ../datasets"

    python3 -m dfn --url http://disk-sophgo-vip.quickconnect.cn/sharing/YlYgQKXqr
    unzip carvana.zip -d ../datasets/
    mv ../datasets/carvana ../datasets/test
    echo "[Success] carvana.zip has been unzipped to path ../datasets/test/"

    python3 -m dfn --url http://disk-sophgo-vip.quickconnect.cn/sharing/2XzlJhbtu
    unzip carvana_masks.zip -d ../datasets/
    mv ../datasets/carvana_masks ../datasets/label
    echo "[Success] carvana_masks has been unzipped to path ../datasets/label/"
   
else
    echo "datasets exist!"
fi


# models
if [ ! -d "../models" ]; 
then
    python3 -m dfn --url http://disk-sophgo-vip.quickconnect.to/sharing/9jxy0fGPf
    ls -al
    unzip models.zip -d ../
    rm models.zip
    echo "models download!"
else
    echo "models exist!"
fi

popd