#!/bin/bash
script_dir=$(dirname "$(readlink -f "$0")")

res=$(which 7z)
if [ $? != 0 ]; then
    echo "Please install 7z on your system!"
    exit
fi

echo "7z is installed in your system!"

pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
python3 -m dfss --url=open@sophgo.com:/sophon-demo/Llama2/models.7z
python3 -m dfss --url=open@sophgo.com:/sophon-demo/Llama2/tools.7z

if [ ! -d "./models" ]; then
    mkdir -p ./models
fi

if [ ! -d "./models/BM1684X" ]; then
    mkdir -p ./models/BM1684X
fi

7z x models.7z -o./models/BM1684X
if [ "$?" = "0" ]; then
  rm models.7z
  echo "Models are ready"
else
  echo "Models unzip error"
fi

7z x tools.7z -o.
if [ "$?" = "0" ]; then
  rm tools.7z
  echo "Tools are ready"
else
  echo "Tools unzip error"
fi

if [ ! -d "./python/token_config" ];
then
    pushd ./python
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Llama2/token_config.7z
    7z x token_config.7z -o.
    rm token_config.7z
    popd
else
    echo "token_config exists! Remove it if you need to update."
fi