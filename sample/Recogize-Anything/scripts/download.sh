#!/bin/bash
scripts_dir=$(dirname $(readlink -f "$0"))

download_bm1684x=0
download_bm1688=0
download_cv186x=0
download_onnx=0
download_ckpt=0

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --BM1684X)
            download_bm1684x=1
            shift 1
            ;;
        --onnx)
            download_onnx=1
            shift 1
            ;;
        --all)
            download_bm1684x=1
            download_bm1688=1
            download_cv186x=1
            download_onnx=1
            download_ckpt=1
            shift 1
            ;;
        *)
            echo "Invalid option: $key" >&2
            exit 1
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            exit 1
            ;;
    esac
done
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
pushd $scripts_dir
# datasets
if [ ! -d "../datasets" ];
then
    mkdir ../datasets
    pushd ../datasets
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Recognize-Anything/test.tar.gz
    tar xvf test.tar.gz && rm test.tar.gz
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Recognize-Anything/cali_set.tar.gz
    tar xvf cali_set.tar.gz && rm cali_set.tar.gz
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Recognize-Anything/ram_tag_list.txt
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Recognize-Anything/ram_tag_list_chinese.txt
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Recognize-Anything/ram_tag_list_threshold.txt
    popd
    echo "datasets download!"
else
    echo "Datasets folder exist! Remove it if you need to update."
fi

# models
if [ ! -d "../models" ]; 
then
    mkdir ../models
fi

pushd ../models

if [ ! -d "../models/BM1684X" ];
then
    if [ $download_bm1684x == 1 ]; then
        python3 -m dfss --url=open@sophgo.com:sophon-demo/Recognize-Anything/BM1684X.tar.gz
        tar xvf BM1684X.tar.gz && rm BM1684X.tar.gz
        python3 -m dfss --url=open@sophgo.com:sophon-demo/Recognize-Anything/BM1684X_INT8.tar.gz
        tar xvf BM1684X_INT8.tar.gz && rm BM1684X_INT8.tar.gz
        echo "models/BM1684X download!"
    fi
else
    echo "models/BM1684X folder exist! Remove it if you need to update."
fi

if [ ! -d "../models/onnx" ];
then
    if [ $download_onnx == 1 ]; then
        python3 -m dfss --url=open@sophgo.com:sophon-demo/Recognize-Anything/ram.onnx
        mkdir ../models/onnx
        mv ram.onnx ../models/onnx
        echo "models/onnx download!"
    fi
else
    echo "models/onnx folder exist! Remove it if you need to update."
fi

popd