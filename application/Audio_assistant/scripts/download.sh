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

if [ ! -d "../models" ]; 
then
    mkdir ../models/
fi

# models
if [ ! -d "../models/BM1688" ]; 
then
    mkdir ../models/BM1688
    python3 -m dfss --url=open@sophgo.com:sophon-demo/application/Audio_assistant/whisper_minicpm_vits_BM1688.zip
    unzip whisper_minicpm_vits_BM1688.zip -d ../models/
    rm whisper_minicpm_vits_BM1688.zip
    echo "models download!"
else
    echo "models/BM1688 folder exist! Remove it if you need to update."
fi

if [ ! -d "../models/BM1684X" ]; 
then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/application/Audio_assistant/whisper_llama3_8b_vits_BM1684X.zip
    unzip whisper_llama3_8b_vits_BM1684X.zip -d ../models
    rm whisper_llama3_8b_vits_BM1684X.zip
    echo "models download!"
else
    echo "models/BM1684X folder exist! Remove it if you need to update."
fi

# datasets
if [ ! -d "../datasets" ]; 
then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/application/Audio_assistant/datasets.zip
    unzip datasets.zip -d ../
    rm datasets.zip
    echo "datasets download!"
else
    echo "datasets folder exist! Remove it if you need to update."
fi

popd
