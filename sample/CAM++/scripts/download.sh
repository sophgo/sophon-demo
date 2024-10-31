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
pushd $scripts_dir

# models
if [ ! -d "../models" ]; then
    mkdir -p ../models
    python3 -m dfss --url=open@sophgo.com:sophon-demo/campplus/models_1023/models.zip
    unzip models.zip -d ../models
    rm models.zip
    echo "models download!"
else
    echo "models folder exist! Remove it if you need to update."
fi

# datasets
if [ ! -d "../datasets" ];
then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/campplus/datasets_1023/datasets.zip
    unzip datasets.zip -d ../
    rm datasets.zip
    echo "datasets download!"
else
    echo "datasets folder exist! Remove it if you need to update."
fi

popd
