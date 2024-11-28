#!/bin/bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
scripts_dir=$(dirname $(readlink -f "$0"))

pushd $scripts_dir

function download_bm1684x {
    if [ ! -d "../models/BM1684X" ]; then
        mkdir -p ../models/BM1684X
    fi
    pushd ../models/BM1684X
        python3 -m dfss --url=open@sophgo.com:sophon-demo/MiniCPM3/minicpm3-4b_int4_seq512_1dev.bmodel
    popd
    pushd ../python/token_config
        python3 -m dfss --url=open@sophgo.com:sophon-demo/MiniCPM3/tokenizer.model
    popd
}


if [ "$1" == "bm1684x" ]; then
    download_bm1684x
else
    echo "Error Parameter"
fi
