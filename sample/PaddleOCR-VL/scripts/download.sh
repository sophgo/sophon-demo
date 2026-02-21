#!/bin/bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
scripts_dir=$(dirname $(readlink -f "$0"))

pushd $scripts_dir
# datasets
if [ ! -d "../datasets" ]; 
then
    pushd ../
    python3 -m dfss --url=open@sophgo.com:sophon-demo/PaddleOCR-VL/datasets.tar.gz
    tar xvf datasets.tar.gz && rm datasets.tar.gz
    popd
    echo "datasets download!"
else
    echo "Datasets folder exist! Remove it if you need to update."
fi

if [ ! -d "../cpp/lib_pcie" ] || [ ! -d "../cpp/lib_soc" ]; 
then
    pushd ../cpp
    python3 -m dfss --url=open@sophgo.com:sophon-demo/PaddleOCR-VL/cpp_libs.tar.gz
    tar xvf cpp_libs.tar.gz && rm cpp_libs.tar.gz
    popd
    echo "cpp libs download!"
else
    echo "cpp libs folder exist! Remove it if you need to update."
fi

if [ ! -d "../configs" ]; 
then
    pushd ../
    python3 -m dfss --url=open@sophgo.com:sophon-demo/PaddleOCR-VL/paddleocr-vl-0.9B/configs.tar.gz
    tar xvf configs.tar.gz && rm configs.tar.gz
    popd
    echo "configs download!"
else
    echo "Configs folder exist! Remove it if you need to update."
fi

# models
if [ ! -d "../models" ]; 
then
    mkdir ../models
    pushd ../models
    python3 -m dfss --url=open@sophgo.com:sophon-demo/PaddleOCR-VL/paddleocr-vl-0.9B/paddleocr-vl_bf16_seq2048_bm1684x_1dev_20260206_230325.bmodel
    python3 -m dfss --url=open@sophgo.com:sophon-demo/PaddleOCR-VL/paddleocr-vl-0.9B/paddleocr-vl_bf16_seq2048_bm1688_1core_static_20260221_195626.bmodel
    popd
    echo "models download!"
else
    echo "Models folder exist! Remove it if you need to update."
fi
popd