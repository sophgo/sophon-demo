#!/bin/bash
scripts_dir=$(dirname $(readlink -f "$0"))
# echo $scripts_dir
pip3 install dfn
pushd $scripts_dir

if [ ! -d '../models' ]; then
    mkdir ../models
    python3 -m dfn --url http://219.142.246.77:65000/sharing/9WvrT9Q58 #c3d bmodels.
    unzip models_*.zip -d ../models
    rm models_*.zip
    cd ../models
    mv models_*/* ../models/
    rm -r models_*
fi


if [ ! -d '../datasets' ]; then
    mkdir ../datasets
    python3 -m dfn --url http://219.142.246.77:65000/sharing/jTuyJMnHO  #ucf 101 test 01
    mv UCF_test_01.tar.gz ../datasets/
    cd ../datasets/
    tar xvf UCF_test_01.tar.gz
    rm UCF_test_01.tar.gz
    python3 -m dfn --url http://219.142.246.77:65000/sharing/86xHDl5rV
    python3 -m dfn --url http://219.142.246.77:65000/sharing/JhtXbNdO1
fi


echo "All done!"
popd
