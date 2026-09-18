#!/bin/bash
# 下载 CV84X6 平台的 bmodel (text_encoder_1/2 F16 + unet/vae BF16)
res=$(which unzip)

if [ $? != 0 ];
then
    echo "Please install unzip on your system!"
    echo "To install, use the following command:"
    echo "sudo apt install unzip"
    exit
fi

pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
scripts_dir=$(dirname $(readlink -f "$0"))

pushd $scripts_dir

# models
if [ ! -d "../models/CV84X6/" ];
then
    mkdir -p ../models/CV84X6/
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Stable_diffusion_XL/CV84X6_bmodels.zip
    unzip CV84X6_bmodels.zip -d ../models/CV84X6/
    rm CV84X6_bmodels.zip

    echo "CV84X6 bmodels download!"
else
    echo "models exists!"
fi

# tokenizer (与 BM1684X 共用)
if [ ! -d "../models/tokenizer" ] || [ ! -d "../models/tokenizer_2" ];
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
