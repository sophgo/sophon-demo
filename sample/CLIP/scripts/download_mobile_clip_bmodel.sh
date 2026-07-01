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

if [ ! -e "../datasets/cali_npz" ];
then
    pushd ../datasets
    python3 -m dfss --url=open@sophgo.com:sophon-demo/CLIP/cali_npz.tar.gz
    tar xvf cali_npz.tar.gz
    rm cali_npz.tar.gz
    popd
    echo "Calibration datasets download!"
else
    echo "Calibration datasets folder exist! Remove it if you need to update."
fi

if [ ! -e "../models" ];
then
    if [ "$target" = "all" ];
    then 
        mkdir -p ../models
        pushd ../models
        python3 -m dfss --url=open@sophgo.com:sophon-demo/CLIP/mobile_clip/BM1684X.zip
        python3 -m dfss --url=open@sophgo.com:sophon-demo/CLIP/mobile_clip/BM1688.zip
        python3 -m dfss --url=open@sophgo.com:sophon-demo/CLIP/mobile_clip/onnx.zip
        unzip BM1684X.zip
        unzip BM1688.zip
        unzip onnx.zip
        rm *.zip

        python3 -m dfss --url=open@sophgo.com:sophon-demo/CLIP/mobile_clip/text_projection_b.npy
        python3 -m dfss --url=open@sophgo.com:sophon-demo/CLIP/mobile_clip/text_projection_blt.npy
        popd
        echo "models download!"
    else
        mkdir -p ../models
        python3 -m dfss --url=open@sophgo.com:sophon-demo/CLIP/mobile_clip/text_projection_b.npy
        python3 -m dfss --url=open@sophgo.com:sophon-demo/CLIP/mobile_clip/text_projection_blt.npy
        mv *.npy ../models

        python3 -m dfss --url=open@sophgo.com:sophon-demo/CLIP/mobile_clip/$target.zip
        python3 -m dfss --url=open@sophgo.com:sophon-demo/CLIP/mobile_clip/onnx.zip
        unzip $target.zip -d ../models
        unzip onnx.zip -d ../models
        rm $target.zip
        rm onnx.zip
        echo "$target models download!"
    fi
else
    echo "Models folder or file exist! Remove it if you need to update."
fi
popd