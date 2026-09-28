#!/bin/bash
res=$(which unzip)
if [ $? != 0 ];
then
    echo "Please install unzip on your system!"
    exit
fi
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade

scripts_dir=$(dirname $(readlink -f "$0"))
pushd $scripts_dir

# models
if [ ! -d "../models/CV84X6" ];
then
    mkdir -p ../models/CV84X6
    pushd ../models/CV84X6
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen2_5_VL/qwen2.5-vl-3b-instruct-awq_w4f16_seq2048_cv84x6_4core_static_20260921_212532.bmodel
    popd
    echo "models download!"
else
    echo "models/CV84X6 folder exist! Remove it if you need to update."
fi

popd
