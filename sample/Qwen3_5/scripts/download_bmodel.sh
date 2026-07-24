#!/bin/bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
scripts_dir=$(dirname $(readlink -f "$0"))

pushd $scripts_dir

function download_datasets {
    if [ ! -d "../datasets" ]; then
        python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen3_5/datasets.zip
        unzip datasets.zip -d ..
        rm datasets.zip
    fi
    pushd ../python
    if [ ! -d "config" ]; then
        python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen3_5/config.zip
        unzip config.zip -d .
        rm config.zip
    fi
    popd
}

function download_bm1684x_2b {
    if [ ! -d "../models/BM1684X" ]; then
        mkdir -p ../models/BM1684X
    fi
    pushd ../models/BM1684X
        python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3.5-2b-int4-autoround_w4bf16_seq2048_bm1684x_1dev_dynamic_20260415_111517.bmodel
        python3 -m dfss --url=open@sophgo.com:/sophon-demo/Qwen3_5/qwen3.5-2b-int4-autoround_w4bf16_seq8192_bm1684x_1dev_history_dynamic_20260722_164018.bmodel
    popd
}

function download_bm1684x_4b {
    if [ ! -d "../models/BM1684X" ]; then
        mkdir -p ../models/BM1684X
    fi
    pushd ../models/BM1684X
        python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3.5-4b-int4-autoround_w4bf16_seq2048_bm1684x_1dev_dynamic_20260416_144422.bmodel
    popd
}

function download_bm1684x_9b {
    if [ ! -d "../models/BM1684X" ]; then
        mkdir -p ../models/BM1684X
    fi
    pushd ../models/BM1684X
        python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3.5-9b-int4-autoround_w4bf16_seq2048_bm1684x_1dev_dynamic_20260416_150658.bmodel
    popd
}

function download_bm1688 {
    if [ ! -d "../models/BM1688" ]; then
        mkdir -p ../models/BM1688
    fi
    pushd ../models/BM1688
        python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3.5-2b-int4-autoround_w4bf16_seq2048_bm1688_2core_dynamic_20260415_212627.bmodel
        python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3.5-4b-int4-autoround_w4bf16_seq2048_bm1688_2core_dynamic_20260416_145112.bmodel
        python3 -m dfss --url=open@sophgo.com:/sophon-demo/Qwen3_5/qwen3.5-2b-int4-autoround_w4bf16_seq8192_bm1688_2core_history_dynamic_20260722_160000.bmodel
    popd
}

if [ "$1" == "bm1684x_2b" ]; then
    download_datasets
    download_bm1684x_2b
elif [ "$1" == "bm1684x_4b" ]; then
    download_datasets
    download_bm1684x_4b
elif [ "$1" == "bm1684x_9b" ]; then
    download_datasets
    download_bm1684x_9b
elif [ "$1" == "bm1688" ]; then
    download_datasets
    download_bm1688
elif [ "$1" == "all" ]; then
    download_datasets
    download_bm1684x_2b
    download_bm1684x_4b
    download_bm1684x_9b
    download_bm1688
else
    echo "Error Parameter"
    echo "Usage: $0 [all|bm1684x_2b|bm1684x_4b|bm1684x_9b|bm1688]"
    exit 1
fi