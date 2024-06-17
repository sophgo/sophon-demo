#!/bin/bash
pip3 install dfss --upgrade

res=$(which unzip)
if [ $? != 0 ];
then
    echo "Please install unzip on your system!"
    echo "To install, use the following command:"
    echo "sudo apt install unzip"
    exit
fi

scripts_dir=$(dirname $(readlink -f "$0"))
# echo $scripts_dir

pushd $scripts_dir

# models
if [ ! -d "../models/BM1684X/" ] || [! -d "../models/onnx_pt"]; 
then
    mkdir -p ../models/BM1684X/
    mkdir -p ../models/onnx_pt/
    python3 -m dfss --url=open@sophgo.com:/sophon-demo/Stable_diffusion_XL/bmodels.zip
    python3 -m dfss --url=open@sophgo.com:/sophon-demo/Stable_diffusion_XL/onnx_pt.zip
    unzip bmodels.zip -d ../models/BM1684X/
    unzip onnx_pt.zip -d ../models/onnx_pt
    rm bmodels.zip
    rm onnx_pt.zip

    echo "bmodels download!"
else
    echo "models exists!"
fi

# tokenizer
if [ ! -d "../models/tokenizer" ] || [ ! -d "../models/tokenizer_2" ] ; 
then
    mkdir -p ../models/tokenizer
    mkdir -p ../models/tokenizer_2
    python3 -m dfss --url=open@sophgo.com:/sophon-demo/Stable_diffusion_XL/tokenizer.zip 
    unzip tokenizer.zip -d ../models/
    rm tokenizer.zip

    echo "tokenizer download!"
else
    echo "tokenizer exists!"
fi
popd