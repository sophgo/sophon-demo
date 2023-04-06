#!/bin/bash
pip3 install dfn
#sudo apt install unzip

scripts_dir=$(dirname $(readlink -f "$0"))
# echo $scripts_dir

pushd $scripts_dir
# datasets
if [ ! -d "../datasets" ]; 
then
    mkdir -p ../datasets/coco
    # test.zip
    python3 -m dfn --url http://219.142.246.77:65000/sharing/0Q12TUQCJ
    unzip test.zip -d ../datasets/
    rm test.zip

    # coco128.zip
    python3 -m dfn --url http://219.142.246.77:65000/sharing/k7TnyA0kS
    unzip coco128.zip -d ../datasets/
    rm coco128.zip

    # dance_1080P.mp4
    python3 -m dfn --url http://219.142.246.77:65000/sharing/ZNVbjs8Am
    mv dance_1080P.mp4 ../datasets/

    # val2017_1000.zip
    python3 -m dfn --url http://219.142.246.77:65000/sharing/rn5EXB0OF
    unzip val2017_1000.zip -d ../datasets/coco
    rm val2017_1000.zip

    # person_keypoints_val2017_1000.json
    python3 -m dfn --url http://219.142.246.77:65000/sharing/7mVCx8bEX
    mv person_keypoints_val2017_1000.json ../datasets/coco

    echo "datasets download!"
else
    echo "datasets exist!"
fi

# models
if [ ! -d "../models" ]; 
then
    python3 -m dfn --url http://219.142.246.77:65000/sharing/7vRbmIrTb
    unzip models.zip -d ../
    rm models.zip
    echo "models download!"
else
    echo "models exist!"
fi
popd