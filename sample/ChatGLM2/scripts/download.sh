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

# models
if [ ! -d "../models" ]; 
then
    mkdir -p ../models/
    mkdir -p ../models/BM1684X
    pushd ../models/BM1684X
    python3 -m dfss --url=open@sophgo.com:sophon-demo/chatglm/models/BM1684X/chatglm2-6b_f16.bmodel
    python3 -m dfss --url=open@sophgo.com:sophon-demo/chatglm/models/BM1684X/chatglm2-6b_int8.bmodel
    python3 -m dfss --url=open@sophgo.com:sophon-demo/chatglm/models/BM1684X/chatglm2-6b_int4.bmodel
    python3 -m dfss --url=open@sophgo.com:sophon-demo/chatglm/models/BM1684X/tokenizer.model
    popd
    echo "models download!"
else
    echo "Models folder exist! Remove it if you need to update."
fi
popd