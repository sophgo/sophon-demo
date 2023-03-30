#!/bin/bash
scripts_dir=$(dirname $(readlink -f "$0"))

res=$(dpkg -l|grep unzip)
if [ $? != 0 ];
then
    echo "Please install unzip on your system!"
    exit
fi
res=$(pip3 list|grep dfn)
if [ $? != 0 ];
then
    pip3 install dfn
fi

pushd $scripts_dir

if [ ! -d '../models' ]; then
    mkdir ../models
    python3 -m dfn --url http://219.142.246.77:65000/sharing/xQh47vOxP #c3d bmodels.
    unzip models.zip -d ../
    rm models.zip
    echo "models download!"
else
    echo "Models folder exist! Remove it if you need to update."
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
else
    echo "Datasets folder exist! Remove it if you need to update."
fi


echo "All done!"
popd
