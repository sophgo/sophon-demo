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
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Seamless/online/streaming_s2t_CV84X6.zip
    unzip streaming_s2t_CV84X6.zip
    mv streaming_s2t_CV84X6/* ../models/CV84X6
    rm -r streaming_s2t_CV84X6
    rm -f streaming_s2t_CV84X6.zip
    echo "streaming bmodel download!"
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Seamless/m4t_offline/m4t_s2t_CV84X6.zip
    unzip m4t_s2t_CV84X6.zip
    mv m4t_s2t_CV84X6/* ../models/CV84X6
    rm -r m4t_s2t_CV84X6
    rm -f m4t_s2t_CV84X6.zip
    echo "m4t bmodel download!"
else
    echo "Models folder exist! Remove it if you need to update."
fi
popd
