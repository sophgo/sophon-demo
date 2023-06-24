#!/bin/bash
pip3 install dfn
# sudo apt install unzip

scripts_dir=$(dirname $(readlink -f "$0"))
# echo $scripts_dir

pushd $scripts_dir
# datasets
if [ ! -d "../datasets/" ];
then
    python3 -m dfn --url http://219.142.246.77:65000/sharing/E6s6xFmzq
    unzip datasets.zip -d ../
    rm datasets.zip

    echo "datasets download!"
else
    echo "Datasets folder exist! Remove it if you need to update."
fi

# models
if [ ! -d "../models" ];
then
    python3 -m dfn --url http://219.142.246.77:65000/sharing/4dgtyCPto
    unzip models.zip -d ../
    rm models.zip
    echo "models download!"
else
    echo "Models folder exist! Remove it if you need to update."
fi
popd