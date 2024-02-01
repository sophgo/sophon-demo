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
if [ ! -d "../models/BM1684X/multilize/" ]; then

    mkdir -p ../models/BM1684X/multilize/
    python3 -m dfss --url=open@sophgo.com:/sophon-demo/Stable_diffusion_v1_5/multilize_bmodels.zip
    unzip multilize_bmodels.zip -d ../models/BM1684X/multilize/
    rm multilize_bmodels.zip

    echo "models_multilize download!"
else
    echo "models_multilize exists!"
fi

# tokenizer
if [ ! -d "../models/tokenizer_path" ]; 
then
    mkdir -p ../models/tokenizer_path
    python3 -m dfss --url=open@sophgo.com:/sophon-demo/Stable_diffusion_v1_5/tokenizer.zip 
    unzip tokenizer.zip -d ../models/tokenizer_path/
    rm tokenizer.zip

    echo "tokenizer download!"
else
    echo "tokenizer exists!"
fi

popd