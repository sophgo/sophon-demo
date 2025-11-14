#!/bin/bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
scripts_dir=$(dirname $(readlink -f "$0"))

pushd $scripts_dir

function download_pics {
    if [ ! -d "../pics" ]; then
        python3 -m dfss --url=open@sophgo.com:sophon-demo/MiniCPMV/pics.zip
        unzip pics.zip -d .
        rm pics.zip
    fi
}

function download_bm1684x {
    if [ ! -d "../models/BM1684X" ]; then
        mkdir -p ../models/BM1684X
    fi
    pushd ../models/BM1684X
        python3 -m dfss --url=open@sophgo.com:sophon-demo/MiniCPMV/BM1684X/minicpm-v-4-awq_w4bf16_seq2048_bm1684x_1dev_20250915_204204.bmodel
    popd

    pushd ../python
    if [ ! -d "token_config" ]; then
        python3 -m dfss --url=open@sophgo.com:sophon-demo/MiniCPM/token_config.zip
        unzip token_config.zip -d .
        rm token_config.zip
    fi
    popd
}

function download_bm1688 {
    if [ ! -d "../models/BM1688" ]; then
        mkdir -p ../models/BM1688
    fi
    pushd ../models/BM1688
        python3 -m dfss --url=open@sophgo.com:sophon-demo/MiniCPMV/BM1688/minicpm-v-4-awq_w4bf16_seq2048_bm1688_2core_20251011_141218.bmodel
    popd

    pushd ../python
    if [ ! -d "token_config" ]; then
        python3 -m dfss --url=open@sophgo.com:sophon-demo/MiniCPM/token_config.zip
        unzip token_config.zip -d .
        rm token_config.zip
    fi
    popd
}

function download_cv186ah {
    if [ ! -d "../models/CV186AH" ]; then
        mkdir -p ../models/CV186AH
    fi
    pushd ../models/CV186AH
        python3 -m dfss --url=open@sophgo.com:sophon-demo/MiniCPMV/CV186AH/minicpm-v-4-awq_w4bf16_seq2048_cv186x_1core_20251011_143349.bmodel
    popd

    pushd ../python
    if [ ! -d "token_config" ]; then
        python3 -m dfss --url=open@sophgo.com:sophon-demo/MiniCPM/token_config.zip
        unzip token_config.zip -d .
        rm token_config.zip
    fi
    popd
}



if [ "$1" == "bm1684x" ]; then
    download_pics
    download_bm1684x
elif [ "$1" == "bm1688" ]; then
    download_pics
    download_bm1688
elif [ "$1" == "cv186ah" ]; then
    download_pics
    download_cv186ah
else
    echo "Error Parameter"
    echo "Usage: $0 [bm1684x|bm1688|cv186ah]"
    exit 1
fi
