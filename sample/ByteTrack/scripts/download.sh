#!/bin/bash
pip3 install dfn
# sudo apt install unzip

# data
if [ ! -d "../data" ];
then
    python3 -m dfn --url http://219.142.246.77:65000/sharing/ckNpogfIK
    unzip data.zip -d ../
    rm data.zip
    echo "data download!"
else
    echo "data exist!"
fi