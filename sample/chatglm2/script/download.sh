#!/bin/bash
pip3 install zip
pip3 install dfn
# sudo apt install unzip

scripts_dir=$(dirname $(readlink -f "$0"))
# echo $scripts_dir

pushd $scripts_dir
# datasets

# models
if [ ! -d "../models" ]; 
then
    python3 -m dfn --url http://disk-sophgo-vip.quickconnect.to/sharing/RYiN4At7v
    unzip models.zip -d ../
    rm models.zip
    echo "models download!"
else
    echo "models exist!"
fi
popd