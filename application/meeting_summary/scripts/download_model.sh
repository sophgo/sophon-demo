#!/bin/bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
scripts_dir=$(dirname $(readlink -f "$0"))

pushd $scripts_dir
# datasets

# models
if [ ! -d "../models" ]; 
then
    mkdir -p ../models/
    pushd ../models/
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen1_5/qwen1.5-7b_int4_6k_1dev.bmodel
    popd
    echo "models download!"
else
    echo "Models folder exist! Remove it if you need to update."
fi
popd