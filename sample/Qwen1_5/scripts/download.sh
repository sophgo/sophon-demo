#!/bin/bash
res=$(which unzip)
if [ $? != 0 ];
then
    echo "Please install unzip on your system!"
    exit
fi
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
scripts_dir=$(dirname $(readlink -f "$0"))

pushd $scripts_dir
# datasets

# models
if [ ! -d "../models" ]; 
then
    mkdir -p ../models/
    mkdir -p ../models/BM1684X
    pushd ../models/BM1684X
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen1_5/qwen1.5-1.8b_int4_1dev.bmodel
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen1_5/qwen1.5-1.8b_int8_1dev.bmodel
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen1_5/qwen1.5-4b_int4_1dev.bmodel
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen1_5/qwen1.5-7b_int4_1dev.bmodel
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen1_5/qwen2-1.5b_int4_seq512_1dev.bmodel
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen1_5/qwen2-1.5b_int8_seq512_1dev.bmodel
    popd
    echo "models download!"
else
    echo "Models folder exist! Remove it if you need to update."
fi

if [ ! -d "../python/token_config" ];
then
    pushd ../python
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen1_5/token_config.zip
    unzip token_config.zip
    rm token_config.zip
    popd
else
    echo "token_config exists! Remove it if you need to update."
fi

popd