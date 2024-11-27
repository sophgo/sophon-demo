#!/bin/bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
scripts_dir=$(dirname $(readlink -f "$0"))

pushd $scripts_dir
# datasets

# models
if [ ! -d "../models" ]; 
then
    mkdir -p ../models/
    pushd ../models
    python3 -m dfss --url=open@sophgo.com:sophon-demo/ChatTTS/chattts-llama_int4_1dev_1024_bm1684x.bmodel
    python3 -m dfss --url=open@sophgo.com:sophon-demo/ChatTTS/chattts-llama_int4_1dev_1024_bm1688.bmodel
    python3 -m dfss --url=open@sophgo.com:sophon-demo/ChatTTS/decoder_1-768-1024_bm1684x.bmodel
    python3 -m dfss --url=open@sophgo.com:sophon-demo/ChatTTS/decoder_1-768-1024_bm1688.bmodel
    python3 -m dfss --url=open@sophgo.com:sophon-demo/ChatTTS/vocos_1-100-2048_bm1684x.bmodel
    python3 -m dfss --url=open@sophgo.com:sophon-demo/ChatTTS/vocos_1-100-2048_bm1688.bmodel
    python3 -m dfss --url=open@sophgo.com:sophon-demo/ChatTTS/asset.tar.gz
    tar xvf asset.tar.gz && rm asset.tar.gz
    popd
    echo "models download!"
else
    echo "Models folder exist! Remove it if you need to update."
fi

popd