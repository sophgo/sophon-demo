#!/bin/bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
scripts_dir=$(dirname $(readlink -f "$0"))

pushd $scripts_dir
# datasets
if [ ! -d "../examples" ]; 
then
    pushd ../
    python3 -m dfss --url=open@sophgo.com:sophon-demo/InternVL3/examples.tar.gz    #test data
    tar xvf examples.tar.gz && rm examples.tar.gz                                   #in case `tar xvf xx` failed.
    popd
    echo "examples download!"
else
    echo "examples folder exist! Remove it if you need to update."

popd
fi