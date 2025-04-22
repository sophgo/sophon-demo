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
if [ ! -d "../models/BM1688" ]; 
then
    mkdir -p ../models/BM1688
    pushd ../models/BM1684X
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen2_5_VL/qwen2.5-vl-3b_bm1688_w4bf16_seq2048.bmodel
    popd
    echo "models download!"
else
    echo "models/BM1688 folder exist! Remove it if you need to update."
fi

popd