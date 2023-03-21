#!/bin/bash
pip3 install dfn
# sudo apt install unzip

# datasets
if [ ! -d "../datasets" ];
then
    python3 -m dfn --url http://219.142.246.77:65000/sharing/GiAmfpnTx
    unzip datasets.zip -d ../
    rm datasets.zip
    echo "datasets download!"
else
    echo "datasets exist!"
fi

# models
if [ ! -d "../models" ];
then
    python3 -m dfn --url http://219.142.246.77:65000/sharing/94cXEPLhS
    unzip models.zip -d ../
    rm models.zip
    echo "models download!"
else
    echo "models exist!"
fi

