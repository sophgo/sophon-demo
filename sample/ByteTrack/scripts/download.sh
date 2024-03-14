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
    python3 -m dfss --url=open@sophgo.com:sophon-demo/ByteTrack/datasets_0918/datasets.zip
    unzip datasets.zip -d ../
    rm datasets.zip

    echo "datasets download!"
else
    echo "Datasets folder exist! Remove it if you need to update."
fi

# models
if [ ! -d "../models" ];
then
    mkdir -p ../models/
    python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv5/models_240314/yolov5s_BM1684.zip
    unzip yolov5s_BM1684.zip -d ../models/
    rm yolov5s_BM1684.zip
    
    python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv5/models_240314/yolov5s_BM1684X.zip
    unzip yolov5s_BM1684X.zip -d ../models/
    rm yolov5s_BM1684X.zip
    
    python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv5/models_240314/yolov5s_BM1688.zip
    unzip yolov5s_BM1688.zip -d ../models/
    rm yolov5s_BM1688.zip
    echo "models download!"
else
    echo "Models folder exist! Remove it if you need to update."
fi
popd