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
if [ ! -d "../BM1688" ]; 
then
    mkdir ../BM1688
    python3 -m dfss --url=open@sophgo.com:sophon-demo/application/Audio_assistant/whisper_minicpm_vits_BM1688.zip
    unzip whisper_minicpm_vits_BM1688.zip -d ../
    rm whisper_minicpm_vits_BM1688.zip
    echo "models download!"
else
    echo "BM1688 folder exist! Remove it if you need to update."
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

# lib
python3 -m dfss --url=open@sophgo.com:sophon-demo/application/Audio_assistant/llama3/support.zip
unzip support.zip -d ../python/Llama3
rm -f support.zip
python3 -m dfss --url=open@sophgo.com:sophon-demo/application/Audio_assistant/minicpm/support.zip
unzip support.zip -d ../python/MiniCPM
rm -f support.zip
python3 -m dfss --url=open@sophgo.com:sophon-demo/application/Audio_assistant/whisper-TPU_py/third_party.zip
unzip third_party.zip -d ../python/whisper-TPU_py/bmwhisper
rm -f third_party.zip
python3 -m dfss --url=open@sophgo.com:sophon-demo/application/Audio_assistant/whisper-TPU_py/assets.zip
unzip assets.zip -d ../python/whisper-TPU_py/bmwhisper
rm -f assets.zip
echo "llama3 minicpm whisper lib update."
popd
