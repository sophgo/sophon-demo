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

function download {
    if [ ! -d "../models/BM1684X" ]; then
        mkdir -p ../models/BM1684X
    fi
    pushd ../models/BM1684X

    if [ x"$1" == x"qwen" ]; then
        python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen/qwen/qwen-7b_int4_seq512_1dev.bmodel
        python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen/qwen/qwen-7b_int4_seq2048_1dev.bmodel 
    elif [ x"$1" == x"qwen1.5" ]; then
        python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen/qwen1.5/qwen1.5-7b_int4_seq512_1dev.bmodel
        python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen/qwen1.5/qwen1.5-7b_int4_seq2048_1dev.bmodel
        python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen/qwen1.5/qwen1.5-7b_int4_seq4096_2dev.bmodel 
    elif [ x"$1" == x"qwen2" ]; then
        python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen/qwen2/qwen2-7b_int4_seq512_1dev.bmodel
    else
        echo "invalie model name"
    fi
    popd
}

download $1
