#!/bin/bash
pip3 install dfss

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
if [ ! -d "../models/BM1684X/singlize/" ]; 
then
    mkdir -p ../models/BM1684X/singlize/
    mkdir -p ../models/onnx_pt/
    python3 -m dfss --url=open@sophgo.com:/sophon-demo/Stable_diffusion_v1_5/singlize_bmodels.zip
    python3 -m dfss --url=open@sophgo.com:/sophon-demo/Stable_diffusion_v1_5/onnx_pt.zip
    unzip singlize_bmodels.zip -d ../models/BM1684X/singlize/
    unzip onnx_pt.zip -d ../models/onnx_pt
    rm singlize_bmodels.zip
    rm onnx_pt.zip

    echo "models_singlize download!"
else
    echo "models_singlize exists!"
fi

# tokenizer
if [ ! -d "../models/tokenizer_path" ]; 
then
    mkdir -p ../models/tokenizer_path
    python3 -m dfss --url=open@sophgo.com:/sophon-demo/Stable_diffusion_v1_5/tokenizer.zip 
else
    echo "tokenizer exists!"
fi
popd