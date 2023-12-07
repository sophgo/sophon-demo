#!/bin/bash

res=$(which unzip)
if [ $? != 0 ];
then
    echo "Please install unzip on your system!"
    exit
fi

echo "unzip is installed in your system!"
 
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
python3 -m dfss --url=open@sophgo.com:/sophon-demo/Llama2/models.zip
python3 -m dfss --url=open@sophgo.com:/sophon-demo/Llama2/tools.zip

if [ ! -d "./models" ]; then
    mkdir -p ./models
fi

if [ ! -d "./models/BM1684X" ]; then
    mkdir -p ./models/BM1684X
fi

unzip -o models.zip -d ./models/BM1684X/
rm models.zip

echo "Models are ready"

unzip -o tools.zip -d .
rm tools.zip

echo "Tools are ready"
 
