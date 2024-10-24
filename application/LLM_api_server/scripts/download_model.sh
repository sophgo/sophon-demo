#!/bin/bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
scripts_dir=$(dirname $(readlink -f "$0"))

pushd $scripts_dir

chip="bm1684x"
if [ "$1" == "bm1688" ]; then
    chip="bm1688"
else
    chip="bm1684x"
fi

# models
if [[ "$chip" == "bm1684x" && ! -d "../models/BM1684X" ]]; then
    mkdir -p ../models/BM1684X
    pushd ../models/BM1684X
    python3 -m dfss --url=open@sophgo.com:sophon-demo/ChatGLM3/models/BM1684X/chatglm3-6b_int4.bmodel
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen/qwen2/qwen2-7b_int4_seq512_1dev.bmodel
    popd
    echo "models download!"
elif [[ "$chip" == "bm1688" && ! -d "../models/BM1688" ]]; then
    mkdir -p ../models/BM1688
    pushd ../models/BM1688
    python3 -m dfss --url=open@sophgo.com:sophon-demo/ChatGLM3/models/BM1688/chatglm3-6b_int4_2core.bmodel
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen/qwen1.5/qwen1.5-1.8b_int4_seq512_bm1688_1dev.bmodel
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen/qwen1.5/qwen1.5-1.8b_int4_seq512_bm1688_1dev_2core.bmodel
    popd
    echo "models download!"
else
    echo "Models folder exist! Remove it if you need to update."
fi

popd
