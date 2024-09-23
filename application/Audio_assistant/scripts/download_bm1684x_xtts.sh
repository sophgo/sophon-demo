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

if [ ! -d "../models/BM1684X" ]; 
then
    mkdir -p ../models/BM1684X
fi

# models
if [ ! -d "../models/BM1684X/xtts" ]; 
then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/application/Audio_assistant/xtts_BM1684X.zip
    unzip xtts_BM1684X.zip -d ../models/BM1684X
    rm xtts_BM1684X.zip
    echo "models download!"
else
    echo "models/BM1684X/xtts folder exist! Remove it if you need to update."
fi

popd