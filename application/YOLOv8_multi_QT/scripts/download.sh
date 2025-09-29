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

# models
if [ ! -d "../models" ]; 
then
    mkdir -p ../models
    python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv8_plus_det/BM1684X.tar.gz    
    tar xvf BM1684X.tar.gz -C ../models
    rm -r BM1684X.tar.gz
    python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv8_plus_det/BM1688.tar.gz
    tar xvf BM1688.tar.gz -C ../models
    rm BM1688.tar.gz
    python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv8_plus_det/CV186X.tar.gz    
    tar xvf CV186X.tar.gz -C ../models
    rm CV186X.tar.gz
    echo "models download!"
else
    echo "Models folder exist! Remove it if you need to update."
fi

# qt
if [ ! -d "../cpp/qt_bm1684x/install" ] || [ ! -d "../cpp/qt_bm1688/install" ];
then
    if [ ! -d "../cpp/qt_bm1684x/install" ]; then
        mkdir -p ../cpp/qt_bm1684x/install
        python3 -m dfss --url=open@sophgo.com:sophon-demo/MultiYolov5/qt-5.14-amd64-aarch64-fl2000fb_v1.1.0.tar.xz
        tar -xf qt-5.14-amd64-aarch64-fl2000fb_v1.1.0.tar.xz --strip-components=1 -C ../cpp/qt_bm1684x/install
        rm -r qt-5.14-amd64-aarch64-fl2000fb_v1.1.0.tar.xz
        echo "Qt for BM1684X downloaded!"
    else
        echo "qt_bm1684x/install folder exists!"
    fi

    if [ ! -d "../cpp/qt_bm1688/install" ]; then
        mkdir -p ../cpp/qt_bm1688/install
        python3 -m dfss --url=open@sophgo.com:sophon-demo/common/qtbase-5.12.8.tar
        tar -xf qtbase-5.12.8.tar --strip-components=1 -C ../cpp/qt_bm1688/install
        rm -r qtbase-5.12.8.tar
        echo "Qt for BM1688 downloaded!"
    else
        echo "qt_bm1688/install folder exists!"
    fi
else
    echo "Qt folders exist! Remove them if you need to update."
fi
popd
