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
if [ ! -d "../datasets" ]; 
then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv5/datasets_0918/datasets.zip
    unzip datasets.zip -d ../
    rm datasets.zip

    echo "datasets download!"
else
    echo "Datasets folder exist! Remove it if you need to update."
fi

if [ ! -d "../models" ]; 
then
    mkdir ../models
    pushd ../models
    python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv8_seg/models/BM1684.tar.gz
    tar xvf BM1684.tar.gz
    python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv8_seg/models/BM1684X.tar.gz
    tar xvf BM1684X.tar.gz
    python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv8_seg/models/BM1688.tar.gz
    tar xvf BM1688.tar.gz
    python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv8_seg/models/onnx.tar.gz
    tar xvf onnx.tar.gz
    python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv8_seg/models/torch.tar.gz
    tar xvf torch.tar.gz
    rm *.tar.gz
    python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv8_seg/models/BM1684X/yolov8s_getmask_32_fp32.bmodel
    mv yolov8s_getmask_32_fp32.bmodel BM1684X/
    python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv8_seg/models/onnx/yolov8s_getmask_32_fp32.onnx
    mv yolov8s_getmask_32_fp32.onnx onnx/

    popd
    echo "models download!"
else
    echo "Models folder exist! Remove it if you need to update."
fi
popd