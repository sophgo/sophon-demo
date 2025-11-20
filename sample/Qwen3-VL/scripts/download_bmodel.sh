#!/bin/bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
scripts_dir=$(dirname $(readlink -f "$0"))

pushd $scripts_dir

function download_datasets {
    if [ ! -d "../datasets" ]; then
        python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen3_VL/datasets.zip
        unzip datasets.zip -d ..
        rm datasets.zip
    fi
    pushd ../python
    if [ ! -d "config" ]; then
        python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen3_VL/config.zip
        unzip config.zip -d .
        rm config.zip
    fi
    popd
}

function download_bm1684x_4b {
    if [ ! -d "../models/BM1684X" ]; then
        mkdir -p ../models/BM1684X
    fi
    pushd ../models/BM1684X
        python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3-vl-4b-instruct_w4bf16_seq2048_bm1684x_1dev_20251026_141347.bmodel
    popd
}

function download_bm1684x_8b {
    if [ ! -d "../models/BM1684X" ]; then
        mkdir -p ../models/BM1684X
    fi
    pushd ../models/BM1684X
        python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3-vl-8b-instruct_w4bf16_seq2048_bm1684x_1dev_20251026_145323.bmodel
    popd
}

function download_bm1688 {
    if [ ! -d "../models/BM1688" ]; then
        mkdir -p ../models/BM1688
    fi
    pushd ../models/BM1688
        python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3-vl-4b-instruct_w4bf16_seq2048_bm1688_2core_20251026_141708.bmodel
    popd
}

if [ "$1" == "bm1684x_4b" ]; then
    download_datasets
    download_bm1684x_4b
elif [ "$1" == "bm1684x_8b" ]; then
    download_datasets
    download_bm1684x_8b
elif [ "$1" == "bm1688" ]; then
    download_datasets
    download_bm1688
elif [ "$1" == "all" ]; then
    download_datasets
    download_bm1684x_4b
    download_bm1684x_8b
    download_bm1688
else
    echo "Error Parameter"
    echo "Usage: $0 [all|bm1684x_4b|bm1684x_8b|bm1688]"
    exit 1
fi
