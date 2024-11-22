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
# datasets

# models
if [ ! $1 ]; then  
    target=all
else
    target=${1^^}

    if [[ $target != "BM1684X" && $target != "BM1688" ]]
        then
        echo "Only support BM1684X, BM1688"
        exit
    fi

fi

if [ ! -e "../models" ];
then
    if [ "$target" = "all" ];
    then 
        python3 -m dfss --url=open@sophgo.com:sophon-demo/CLIP/cn_clip/models.zip
        unzip models.zip -d ../
        rm models.zip
        echo "models download!"
    else
        mkdir -p ../models
        python3 -m dfss --url=open@sophgo.com:sophon-demo/CLIP/cn_clip/$target.zip
        unzip $target.zip -d ../models
        rm $target.zip
        echo "$target models download!"
    fi
else
    echo "Models folder or file exist! Remove it if you need to update."
fi
popd