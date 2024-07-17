#!/bin/bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
scripts_dir=$(dirname $(readlink -f "$0"))

pushd $scripts_dir
# datasets
if [ ! -d "../datasets" ]; 
then
    mkdir ../datasets
    pushd ../datasets
    python3 -m dfss --url=open@sophgo.com:sophon-demo/common/test.tar.gz    #test pictures
    tar xvf test.tar.gz && rm test.tar.gz                                   #in case `tar xvf xx` failed.
    python3 -m dfss --url=open@sophgo.com:sophon-demo/common/coco.names     #coco classnames
    python3 -m dfss --url=open@sophgo.com:sophon-demo/common/coco128.tar.gz #coco 128 pictures
    tar xvf coco128.tar.gz && rm coco128.tar.gz
    python3 -m dfss --url=open@sophgo.com:sophon-demo/common/coco_val2017_1000.tar.gz #coco 1000 pictures and json.
    tar xvf coco_val2017_1000.tar.gz && rm coco_val2017_1000.tar.gz
    python3 -m dfss --url=open@sophgo.com:sophon-demo/common/test_car_person_1080P.mp4 #test video
    popd
    echo "datasets download!"
else
    echo "Datasets folder exist! Remove it if you need to update."
fi

# models
if [ ! -d "../models" ]; 
then
    mkdir ../models
    pushd ../models
    python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv8_det/models/BM1684.tar.gz
    tar xvf BM1684.tar.gz && rm BM1684.tar.gz
    python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv8_det/models/BM1684X.tar.gz
    tar xvf BM1684X.tar.gz && rm BM1684X.tar.gz
    python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv8_det/models/BM1688.tar.gz
    tar xvf BM1688.tar.gz && rm BM1688.tar.gz
    python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv8_det/models/CV186X.tar.gz
    tar xvf CV186X.tar.gz && rm CV186X.tar.gz
    python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv8_det/models/onnx.tar.gz
    tar xvf onnx.tar.gz && rm onnx.tar.gz
    popd
    echo "models download!"
else
    echo "Models folder exist! Remove it if you need to update."
fi
popd
