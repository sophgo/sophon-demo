#!/bin/bash
pip3 install dfn
# sudo apt install unzip

scripts_dir=$(dirname $(readlink -f "$0"))
# echo $scripts_dir

pushd $scripts_dir
# datasets
if [ ! -d "../datasets" ]; 
then
    python3 -m dfn --url http://219.142.246.77:65000/sharing/BYL5yfRm9
    7z x datasets.zip -o../
    rm datasets.zip

    echo "datasets download!"
else
    echo "datasets exist!"
fi

# models
if [ ! -d "../models" ]; 
then
    python3 -m dfn --url http://219.142.246.77:65000/sharing/gFbcS2Ftw
    unzip models.zip -d ../
    rm models.zip
    echo "models download!"
else
    echo "models exist!"
fi
popd