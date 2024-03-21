#!/bin/bash
res=$(dpkg -l|grep unzip)
if [ $? != 0 ];
then
    echo "Please install unzip on your system!"
    exit
fi
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
scripts_dir=$(dirname $(readlink -f "$0"))

pushd $scripts_dir

# models
if [ ! -d "../models" ]; 
then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv5_opt/models_0918/models.zip
    unzip models.zip -d ../
    rm models.zip
    rm ../models/BM1684X/yolov5s_tpukernel_int8_4b.bmodel
    echo "models download!"
else
    echo "Models folder exist! Remove it if you need to update."
fi

# lib
if [ ! -d "../tools/lib" ]; 
then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/MultiYolov5/lib_1011.tar.gz
    tar -zxvf lib_1011.tar.gz -C ../tools/
    rm lib_1011.tar.gz
    echo "lib download!"
else
    echo "lib folder exist! Remove it if you need to update."
fi

# qt install
if [ ! -d "../tools/install" ]; 
then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/MultiYolov5/qt-5.14-amd64-aarch64-fl2000fb_v1.1.0.tar.xz
    tar -xaf qt-5.14-amd64-aarch64-fl2000fb_v1.1.0.tar.xz -C ../tools/
    rm qt-5.14-amd64-aarch64-fl2000fb_v1.1.0.tar.xz
    echo "qt download!"
else
    echo "install folder exist! Remove it if you need to update."
fi

popd