#!/bin/bash
res=$(dpkg -l|grep unzip)
if [ $? != 0 ];
then
    echo "Please install unzip on your system!"
    exit
fi
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
scripts_dir=$(dirname $(readlink -f "$0"))

pushd $scripts_dir
# datasets
if [ ! -d "../datasets" ]; 
then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/PP-OCR/datasets_0918/datasets.zip
    unzip datasets.zip -d ../
    rm datasets.zip

    echo "datasets download!"
else
    echo "Datasets folder exist! Remove it if you need to update."
fi

# models
if [ ! -d "../models" ]; 
then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/PP-OCR/models_0918/models.zip
    unzip models.zip -d ../
    rm models.zip
    rm ../models/BM1684/ch_PP-OCRv3_*_int8.bmodel ../models/BM1684X/ch_PP-OCRv3_*_int8.bmodel #int8 has bad performance, if you want to test it, you can comment this line
    pushd ../models/
    python3 -m dfss --url=open@sophgo.com:sophon-demo/PP-OCR/models_231110/BM1688.zip
    unzip BM1688.zip
    rm -r BM1688.zip
    popd
    echo "models download!"
else
    echo "Models folder exist! Remove it if you need to update."
fi
popd