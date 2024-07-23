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
if [ ! -d "../models" ]; 
then
    mkdir ../models
    mkdir ../models/BM1684X
    python3 -m dfss --url=open@sophgo.com:sophon-demo/application/Grounded-sam/groundingdino.zip
    unzip groundingdino.zip -d ../models/BM1684X
    rm groundingdino.zip

    python3 -m dfss --url=open@sophgo.com:sophon-demo/application/Grounded-sam/embedding_bmodel.zip
    unzip embedding_bmodel.zip -d ../models/BM1684X
    rm embedding_bmodel.zip

    python3 -m dfss --url=open@sophgo.com:sophon-demo/application/Grounded-sam/decode_bmodel.zip
    unzip decode_bmodel.zip -d ../models/BM1684X
    rm decode_bmodel.zip

    python3 -m dfss --url=open@sophgo.com:sophon-demo/GroundingDINO/bert-base-uncased.zip
    unzip bert-base-uncased.zip -d ../models
    rm bert-base-uncased.zip

    echo "models download!"
else
    echo "Models folder exist! Remove it if you need to update."
fi
popd
