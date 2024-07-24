#!/bin/bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
script_dir=$(dirname $(readlink -f "$0"))
echo $script_dir

pushd $script_dir

# datasets
if [ ! -d "../datasets" ];
then
    mkdir ../datasets
    pushd ../datasets
    python3 -m dfss --url=open@sophgo.com:sophon-demo/segformer/datasets/cali.tar.gz                       
    tar xvf cali.tar.gz && rm cali.tar.gz                                   
    python3 -m dfss --url=open@sophgo.com:sophon-demo/segformer/datasets/cityscapes.tar.gz    
    tar xvf cityscapes.tar.gz && rm cityscapes.tar.gz
    python3 -m dfss --url=open@sophgo.com:sophon-demo/segformer/datasets/cityscapes_small.tar.gz 
    tar xvf cityscapes_small.tar.gz && rm cityscapes_small.tar.gz
    python3 -m dfss --url=open@sophgo.com:sophon-demo/segformer/datasets/test_car_person_1080P.mp4         
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
    python3 -m dfss --url=open@sophgo.com:sophon-demo/segformer/models/BM1684.tar.gz    
    tar xvf BM1684.tar.gz && rm BM1684.tar.gz 
    python3 -m dfss --url=open@sophgo.com:sophon-demo/segformer/models/BM1684X.tar.gz    
    tar xvf BM1684X.tar.gz && rm BM1684X.tar.gz 
    python3 -m dfss --url=open@sophgo.com:sophon-demo/segformer/models/BM1688.tar.gz     
    tar xvf BM1688.tar.gz  && rm BM1688.tar.gz  
    python3 -m dfss --url=open@sophgo.com:sophon-demo/segformer/models/CV186X.tar.gz    
    tar xvf CV186X.tar.gz && rm CV186X.tar.gz 
    python3 -m dfss --url=open@sophgo.com:sophon-demo/segformer/models/onnx.tar.gz    
    tar xvf onnx.tar.gz && rm onnx.tar.gz 
    popd
    echo "models download!"
else
    echo "Models folder exist! Remove it if you need to update."
fi
popd
