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

# models
if [ ! -d "../models/BM1684X" ]; 
then
    mkdir -p ../models/BM1684X
    pushd ../models/BM1684X
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen2-VL/qwen2-vl-7b_int4_seq512_1dev.bmodel
    popd
    echo "models download!"
else
    echo "models/BM1684X folder exist! Remove it if you need to update."
fi

popd