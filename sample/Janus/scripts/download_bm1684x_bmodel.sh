#!/bin/bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
scripts_dir=$(dirname $(readlink -f "$0"))
pushd $scripts_dir

# models
if [ ! -d "../models" ]; 
then
    mkdir -p ../models/BM1684X
    pushd ../models/BM1684X
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Janus/janus-pro-7b_int4_seq2048.bmodel
    popd
    echo "models download!"
else
    echo "models folder exist! Remove it if you need to update."
fi

popd