#!/bin/bash

res=$(dpkg -l|grep unzip)
if [ $? != 0 ];
then
    echo "Please install unzip on your system!"
    exit
fi

res=$(which 7z)
if [ $? != 0 ];
then
    echo "Please install 7z on your system!"
    echo "To install, use the following command:"
    echo "sudo apt install p7zip;sudo apt install p7zip-full"
    exit
fi

pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
scripts_dir=$(dirname $(readlink -f "$0"))

pushd $scripts_dir
# datasets
if [ ! -d "../datasets" ]; 
then
    mkdir -p ../datasets
    python3 -m dfss --url=open@sophgo.com:/sophon-demo/license_plate_recognition/1080_1920_5s.zip
    unzip 1080_1920_5s.zip 
    mv 1080_1920_5s.mp4 ../datasets/   
    rm -f 1080_1920_5s.zip
    python3 -m dfss --url=open@sophgo.com:/sophon-demo/license_plate_recognition/licenseplate_640516-h264.7z
    7z x licenseplate_640516-h264.7z 
    mv licenseplate_640516-h264.mp4 ../datasets/  
    rm -f licenseplate_640516-h264.7z 

    echo "datasets download!"
else
    echo "Datasets folder exist! Remove it if you need to update."
fi

# models
if [ ! -d "../models" ]; 
then
    mkdir -p ../models ../models/lprnet/
    python3 -m dfss --url=open@sophgo.com:/sophon-stream/lprnet/models.7z    
    7z x models.7z 
    mv models/* ../models/lprnet/
    rm -r models.7z models

    python3 -m dfss --url=open@sophgo.com:/sophon-stream/license_plate_recognition/yolov5s-licensePlate.7z
    7z x yolov5s-licensePlate.7z -o../models
    rm yolov5s-licensePlate.7z

    python3 -m dfss --url=open@sophgo.com:/sophon-demo/license_plate_recognition/yolov5_licenseplate/CV186X.zip
    unzip ./CV186X.zip
    mkdir -p ../models/yolov5s-licensePlate/CV186X
    mv ./CV186X/yolov5* ../models/yolov5s-licensePlate/CV186X
    rm -rf ./CV186X.zip
    rm -rf ./BM1688


    python3 -m dfss --url=open@sophgo.com:/sophon-demo/LPRNet/BM1688.zip
    unzip ./BM1688.zip
    mkdir -p ../models/lprnet/BM1688
    mv ./BM1688/lprnet* ../models/lprnet/BM1688
    rm -rf ./BM1688.zip
    rm -rf ./BM1688

    python3 -m dfss  --url=open@sophgo.com:sophon-demo/LPRNet/models_240328/CV186X.zip
    unzip ./CV186X.zip
    mkdir -p ../models/lprnet/CV186X
    mv ./CV186X/lprnet* ../models/lprnet/CV186X
    rm -rf ./CV186X.zip
    rm -rf ./CV186X

    python3 -m dfss --url=open@sophgo.com:/sophon-demo/YOLOv5/license_plate_recognition/BM1688.zip
    unzip ./BM1688.zip
    mkdir -p ../models/yolov5s-licensePlate/BM1688
    mv ./BM1688/yolov5s* ../models/yolov5s-licensePlate/BM1688
    rm -rf ./BM1688.zip
    rm -rf ./BM1688

    echo "models download!"
else
    echo "Models folder exist! Remove it if you need to update."
fi
popd