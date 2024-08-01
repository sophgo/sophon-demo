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
    python3 -m dfss --url=open@sophgo.com:sophon-demo/ChatGLM4/models/BM1684X/glm4-9b_int4_1dev.bmodel
    python3 -m dfss --url=open@sophgo.com:sophon-demo/ChatGLM4/models/BM1684X/glm4-9b_int8_1dev.bmodel
    popd
    echo "models download!"
else
    echo "Models folder exist! Remove it if you need to update."
fi

if [ ! -d "../python/token_config" ];
then
    pushd ../python
    python3 -m dfss --url=open@sophgo.com:sophon-demo/ChatGLM4/token_config.zip
    unzip token_config.zip
    rm token_config.zip
    popd
else
    echo "token_config exists! Remove it if you need to update."
fi

popd