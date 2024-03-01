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
    python3 -m dfss --url=open@sophgo.com:sophon-demo/ChatGLM3/models/BM1684X/chatglm3-6b_fp16.bmodel
    python3 -m dfss --url=open@sophgo.com:sophon-demo/ChatGLM3/models/BM1684X/chatglm3-6b_int8.bmodel
    python3 -m dfss --url=open@sophgo.com:sophon-demo/ChatGLM3/models/BM1684X/chatglm3-6b_int4.bmodel
    popd
    echo "models download!"
else
    echo "Models folder exist! Remove it if you need to update."
fi

if [ ! -d "../python/token_config" ];
then
    pushd ../python
    python3 -m dfss --url=open@sophgo.com:sophon-demo/ChatGLM3/token_config.zip
    unzip token_config.zip
    rm token_config.zip
    popd
else
    echo "token_config exists! Remove it if you need to update."
fi

popd