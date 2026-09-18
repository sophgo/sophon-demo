#!/bin/bash
# 下载 CV84X6 平台的 singlize bmodel (text_encoder F16 + unet/vae BF16)
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
if [ ! -d "../models/CV84X6/singlize/" ];
then
    mkdir -p ../models/CV84X6/singlize/
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Stable_diffusion_v1_5/CV84X6_singlize_bmodels.zip
    unzip CV84X6_singlize_bmodels.zip -d ../models/CV84X6/singlize/
    rm CV84X6_singlize_bmodels.zip

    echo "CV84X6 singlize bmodels download!"
else
    echo "models_singlize exists!"
fi

# tokenizer (与 BM1684X 共用)
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
