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
    python3 -m dfn --url http://219.142.246.77:65000/sharing/Rul34bKDI
    mv coco.names ../datasets

    # test.zip
    python3 -m dfn --url http://219.142.246.77:65000/sharing/kHaKmNrSm
    unzip test.zip -d ../datasets/
    rm test.zip

    # test_car_person_1080P.mp4
    python3 -m dfn --url http://219.142.246.77:65000/sharing/XyAx8A51K
    mv test_car_person_1080P.mp4 ../datasets

    # coco128.zip
    python3 -m dfn --url http://219.142.246.77:65000/sharing/k7TnyA0kS
    unzip coco128.zip -d ../datasets/
    rm coco128.zip

    # val2017.zip
    python3 -m dfn --url http://219.142.246.77:65000/sharing/np8j6kg3b
    unzip val2017.zip -d ../datasets/coco
    rm val2017.zip

    # instances_val2017.json
    python3 -m dfn --url http://219.142.246.77:65000/sharing/OpUgYWuVg
    mv instances_val2017.json ../datasets/coco

    echo "datasets download!"
else
    echo "datasets exist!"
fi

# models
if [ ! -d "../models" ]; 
then
    python3 -m dfn --url http://219.142.246.77:65000/sharing/Z59yeHxl2
    unzip models.zip -d ../
    rm models.zip
    echo "models download!"
else
    echo "models exist!"
fi
popd