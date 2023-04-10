#!/bin/bash
pip3 install dfn
# sudo apt install unzip

scripts_dir=$(dirname $(readlink -f "$0"))
# echo $scripts_dir

pushd $scripts_dir
# datasets
if [ ! -d "../datasets" ]; 
then
    mkdir -p ../datasets/coco
    # coco.names
    python3 -m dfn --url http://219.142.246.77:65000/sharing/FzOejP895
    mv coco.names ../datasets

    # test.zip
    python3 -m dfn --url http://219.142.246.77:65000/sharing/n1gCAKCOU
    unzip test.zip -d ../datasets/
    rm test.zip

    # centernet test picture
    python3 -m dfn --url http://219.142.246.77:65000/sharing/evXTVV1Af
    mv ctdet_test.jpg ../datasets
    echo "[Success] ctdet_test.jpg has been downloaded to path ../datasets"


    # coco128.zip
    python3 -m dfn --url http://219.142.246.77:65000/sharing/LNlCFMiz5
    unzip coco128.zip -d ../datasets/
    rm coco128.zip

    # val2017_1000.zip
    python3 -m dfn --url http://219.142.246.77:65000/sharing/J4VGhNhGu
    unzip val2017_1000.zip -d ../datasets/coco
    rm val2017_1000.zip

    # instances_val2017_1000.json
    python3 -m dfn --url http://219.142.246.77:65000/sharing/rXe4WKCTd
    mv instances_val2017_1000.json ../datasets/coco

    echo "datasets download!"
else
    echo "datasets exist!"
fi

# models
if [ ! -d "../models" ]; 
then
    python3 -m dfn --url http://219.142.246.77:65000/sharing/WCI0q1dPy
    unzip models.zip -d ../
    rm models.zip
    
    echo "models download!"
else
    echo "models exist!"
fi
popd