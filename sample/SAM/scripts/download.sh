#!/bin/bash
res=$(which unzip)
if [ $? != 0 ];
then
    echo "Please install unzip on your system!"
    exit
fi

if [ ! $2 ]; then
    model_type=SAM-ViT-B
else
    model_type=$2
fi

pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
scripts_dir=$(dirname $(readlink -f "$0"))

pushd $scripts_dir
# datasets
if [ ! -d "../datasets" ]; 
then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/SAM/datasets.zip
    unzip datasets.zip -d ../datasets
    rm datasets.zip

    echo "datasets download!"
else
    echo "Datasets folder exist! Remove it if you need to update."
fi

# models
if [ ! -d "../models" ]; 
then
    echo "downloading model_type: $model_type."
    if test $model_type = "SAM-ViT-B"; then
        python3 -m dfss --url=open@sophgo.com:sophon-demo/SAM/models.zip
        unzip models.zip -d ../
        rm models.zip
    elif test $model_type = "SAM-ViT-T"; then
        python3 -m dfss --url=open@sophgo.com:sophon-demo/SAM/mobile_models.tar.gz
        tar xvf mobile_models.tar.gz -C ../ && rm mobile_models.tar.gz
    else
        echo "unsupport model_type: $model_type, only supports: SAM-ViT-B,SAM-ViT-T."
    fi
    echo "models download!"
else
    echo "Models folder exist! Remove it if you need to update."
fi
popd
