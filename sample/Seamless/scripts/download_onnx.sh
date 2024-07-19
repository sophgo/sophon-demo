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
if [ ! -d "../models/onnx" ];
then
    mkdir -p ../models/onnx
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Seamless/online/streaming_s2t_onnx.zip
    unzip streaming_s2t_onnx.zip -d ../models/onnx
    rm streaming_s2t_onnx.zip
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Seamless/m4t_offline/m4t_s2t_onnx.zip
    unzip m4t_s2t_onnx.zip -d ../models/onnx
    rm m4t_s2t_onnx.zip
    echo "onnx models download!"
else
    echo "Models folder exist! Remove it if you need to update."
fi
popd