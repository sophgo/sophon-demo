#!/bin/bash
script_dir=$(dirname "$(readlink -f "$0")")
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
pushd $script_dir
if [ ! -d "../models/BM1684X" ]; then
    mkdir -p ../models/BM1684X
    pushd ../models/BM1684X
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Llama2/llama_w4f16_seq512_20250121_171104.tar.gz
    tar xvf llama_w4f16_seq512_20250121_171104.tar.gz && rm llama_w4f16_seq512_20250121_171104.tar.gz
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Llama2/tokenizer.tar.gz
    tar xvf tokenizer.tar.gz && rm tokenizer.tar.gz
    mv tokenizer llama_w4f16_seq512_20250121_171104
    popd
else
    echo "./models/BM1684X exists, remove it if you want to update your models."
fi
popd