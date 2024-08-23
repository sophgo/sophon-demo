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
python3 -m dfss --url=open@sophgo.com:sophon-demo/BLIP/datasets.zip
unzip datasets.zip -d ..
rm datasets.zip

# models
if [ ! $1 ]; then  
    target=all
else
    target=${1^^}

    if [[ $target != "BM1684" && $target != "BM1684X" && $target != "BM1688" ]]
        then
        echo "Only support BM1684, BM1684X, BM1688"
        exit
    fi

fi

function download_target()
{
    python3 -m dfss --url=open@sophgo.com:sophon-demo/BLIP/$1.zip
    unzip $1.zip -d ../models
    rm $1.zip
    echo "$1 models download!"
}

if [ ! -e "../models" ];
then
    mkdir -p ../models
    download_target bert-base-uncased
    if [ "$target" = "all" ];
    then 
        for target in BM1684 BM1684X BM1688
        do
            download_target $target
        done
    else
        download_target $target
    fi
else
    echo "Models folder or file exist! Remove it if you need to update."
fi
popd
