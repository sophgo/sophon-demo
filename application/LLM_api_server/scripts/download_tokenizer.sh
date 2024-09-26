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

echo $scripts_dir
pushd ../python/utils/chatglm3
python3 -m dfss --url=open@sophgo.com:sophon-demo/ChatGLM3/token_config.zip
unzip token_config.zip
rm token_config.zip
popd

pushd ../python/utils/qwen
python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen1_5/token_config.zip
unzip token_config.zip
rm token_config.zip
popd

popd