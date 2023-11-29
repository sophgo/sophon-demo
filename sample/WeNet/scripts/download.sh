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
    mkdir -p ../datasets
    python3 -m dfss --url=open@sophgo.com:sophon-demo/WeNet/datasets/aishell_S0764.zip
    unzip aishell_S0764.zip -d ../datasets/
    rm aishell_S0764.zip

    echo "datasets download!"
else
    echo "Datasets folder exist! Remove it if you need to update."
fi

# models
if [ ! -d "../models" ]; 
then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/WeNet/models_231129/models.zip
    unzip models.zip -d ../
    rm models.zip
    echo "models download!"
else
    echo "Models folder exist! Remove it if you need to update."
fi

# swig decoders module
if [ ! -d "../python/swig_decoders" ]; 
then
    if [ "$1" = "soc" ];
    then
        python3 -m dfss --url=open@sophgo.com:sophon-demo/WeNet/soc/swig_decoders_aarch64.zip
        unzip swig_decoders_aarch64.zip -d ../python/
        mv ../python/swig_decoders_aarch64 ../python/swig_decoders
        rm swig_decoders_aarch64.zip
    elif [ "$1" = "pcie" ];
    then
        python3 -m dfss --url=open@sophgo.com:sophon-demo/WeNet/pcie/swig_decoders_x86_64.zip
        unzip swig_decoders_x86_64.zip -d ../python/
        mv ../python/swig_decoders_x86_64 ../python/swig_decoders
        rm swig_decoders_x86_64.zip
    else
        echo "Please set the platform. Option: soc and pcie"
        exit
    fi

    echo "swig decoders module download!"
else
    echo "swig decoders module exist!"
fi
popd