#!/bin/bash



pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade

scripts_dir=$(dirname $(readlink -f "$0"))

pushd $scripts_dir
# datasets
if [ ! -d "../data" ]; 
then
    python3 -m dfss --url=open@sophgo.com:/sophon-stream/dwa_blend_encode/data.tar.gz
    tar -zxvf data.tar.gz
    mv data ..
    rm -f data.tar.gz
    python3 -m dfss --url=open@sophgo.com:/sophon-stream/gdwa_blend_encode/data.zip
    unzip data.zip
    cp -rf data ..
    rm -f data
    rm -f data.zip

else
    echo "test image exist!"
fi

popd