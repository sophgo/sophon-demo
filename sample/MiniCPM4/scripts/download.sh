#!/bin/bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
scripts_dir=$(dirname $(readlink -f "$0"))

pushd $scripts_dir

function download_bm1684x {
    if [ ! -d "../models/BM1684X" ]; then
        mkdir -p ../models/BM1684X
    fi
    pushd ../models/BM1684X
        python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/minicpm4-8b_w4bf16_seq512_bm1684x_1dev_20250613_175044.bmodel
        python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/minicpm4-8b_w4bf16_seq8192_bm1684x_1dev_20250613_182940.bmodel
    popd

    pushd ../python/token_config
        python3 -m dfss --url=open@sophgo.com:sophon-demo/MiniCPM4/tokenizer.model
    popd
}

function download_bm1688 {
    if [ ! -d "../models/BM1688" ]; then
        mkdir -p ../models/BM1688
    fi
    pushd ../models/BM1688
        # minicpm4-0.5b bm1688 512
        python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU_Lite/minicpm4-0.5b-gptq_w4bf16_seq512_bm1688_2core_20250616_122001.bmodel
    popd

    pushd ../python/token_config
        python3 -m dfss --url=open@sophgo.com:sophon-demo/MiniCPM4/tokenizer.model
    popd
}

function download_cv186ah {
    if [ ! -d "../models/CV186AH" ]; then
        mkdir -p ../models/CV186AH
    fi
    pushd ../models/CV186AH
        # minicpm4-0.5b cv186ah 512
        python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU_Lite/minicpm4-0.5b-gptq_w4bf16_seq512_cv186x_1core_20250616_122126.bmodel 
    popd

    pushd ../python/token_config
        python3 -m dfss --url=open@sophgo.com:sophon-demo/MiniCPM4/tokenizer.model
    popd
}



if [ "$1" == "bm1684x" ]; then
    download_bm1684x
elif [ "$1" == "bm1688" ]; then
    download_bm1688
elif [ "$1" == "cv186ah" ]; then
    download_cv186ah
else
    echo "Error Parameter"
    echo "Usage: $0 [bm1684x|bm1688|cv186ah]"
    exit 1
fi
