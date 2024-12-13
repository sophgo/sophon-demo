#!/bin/bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
scripts_dir=$(dirname $(readlink -f "$0"))

pushd $scripts_dir
# datasets
if [ ! -d "../datasets" ]; 
then
    mkdir ../datasets
    pushd ../datasets
    python3 -m dfss --url=open@sophgo.com:sophon-demo/common/test_car_person_1080P.mp4 #test video
    popd
    echo "datasets download!"
else
    echo "Datasets folder exist! Remove it if you need to update."
fi
# models
if [ ! -d "../models" ]; 
then
    mkdir -p ../models/BM1684X
    pushd ../models/BM1684X
    python3 -m dfss --url=open@sophgo.com:sophon-demo/vila/vision_embedding_1batch.bmodel
    python3 -m dfss --url=open@sophgo.com:sophon-demo/vila/vision_embedding_6batch.bmodel
    python3 -m dfss --url=open@sophgo.com:sophon-demo/vila/llama_int4_seq2560.bmodel
    popd
    pushd ../models
    python3 -m dfss --url=open@sophgo.com:sophon-demo/vila/BM1688.tgz
    tar zxvf BM1688.tgz
    rm -rf BM1688.tgz
    popd
    echo "models download!"
else
    echo "Models folder exist! Remove it if you need to update."
fi
popd