#!/bin/bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
scripts_dir=$(dirname $(readlink -f "$0"))

pushd $scripts_dir
# datasets

# models
if [ ! -d "../models/BM1684X" ]; 
then
    mkdir -p ../models/BM1684X
    pushd ../models/BM1684X
    python3 -m dfss --url=open@sophgo.com:sophon-demo/ChatGLM3/models/BM1684X/chatglm3-6b_int4.bmodel
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen/qwen2/qwen2-7b_int4_seq512_1dev.bmodel
    popd
    echo "models download!"
else
    echo "Models folder exist! Remove it if you need to update."
fi

popd
