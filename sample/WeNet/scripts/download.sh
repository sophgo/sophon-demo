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
    unzip aishell_S0764.zip -d ../datasets
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

if [ ! -d "../cpp/cross_compile_module" ];
then
    mkdir -p ../cpp/cross_compile_module
    pushd ../cpp/cross_compile_module
    python3 -m dfss --url=open@sophgo.com:sophon-demo/WeNet/soc/3rd_party.tar.gz
    python3 -m dfss --url=open@sophgo.com:sophon-demo/WeNet/soc/ctcdecode-cpp.tar.gz
    tar zxf 3rd_party.tar.gz
    tar zxf ctcdecode-cpp.tar.gz
    rm 3rd_party.tar.gz
    rm ctcdecode-cpp.tar.gz
    popd
    echo "cross_compile_module download!"
else
    echo "cross_compile_module exist, please remove it if you want to update."
fi

# swig decoders module
if [ ! -d "../python/swig_decoders_aarch64" ]; 
then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/WeNet/soc/swig_decoders_aarch64.zip
    unzip swig_decoders_aarch64.zip -d ../python/
    rm swig_decoders_aarch64.zip
    echo "swig_decoders_aarch64 download!"
else
    echo "swig_decoders_aarch64 exist, please remove it if you want to update."
fi

if [ ! -d "../python/swig_decoders_x86_64" ]; 
then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/WeNet/pcie/swig_decoders_x86_64.zip
    unzip swig_decoders_x86_64.zip -d ../python/
    rm swig_decoders_x86_64.zip
    echo "swig_decoders_x86_64 download!"
else
    echo "swig_decoders_x86_64 exist, please remove it if you want to update."
fi



popd