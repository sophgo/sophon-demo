#!/bin/bash
res=$(which unzip)
if [ $? != 0 ]; then
    echo "Please install unzip on your system!"
    echo "Please run the following command: sudo apt-get install unzip"
    exit
fi
echo "unzip is installed in your system!"

pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade

scripts_dir=$(dirname $(readlink -f "$0"))
name=speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404
pushd $scripts_dir

# models
if [ ! -d "../models" ]; then
    mkdir -p ../models
    python3 -m dfss --url=open@sophgo.com:sophon-demo/FunASR/${name}/BM1684X.zip
    unzip BM1684X.zip -d ../models
    rm BM1684X.zip
    echo "BM1684X model download!"
else
    echo "models folder exist! Remove it if you need to update."
fi

popd
