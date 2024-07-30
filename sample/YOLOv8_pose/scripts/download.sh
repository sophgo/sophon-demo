#!/bin/bash
pip3 install dfss

scripts_dir=$(dirname $(readlink -f "$0"))
# echo $scripts_dir

pushd $scripts_dir
# datasets
if [ ! -d "../datasets" ]; 
then
    mkdir -p ../datasets/coco
    # test.zip
    python3 -m dfss --url=open@sophgo.com:sophon-demo/common/test_pose.tar.gz
    tar -zxf test_pose.tar.gz -C ../datasets/
    rm test_pose.tar.gz

    # coco128.zip
    python3 -m dfss --url=open@sophgo.com:sophon-demo/common/coco128.tar.gz
    tar -zxf coco128.tar.gz -C ../datasets/
    rm coco128.tar.gz

    # dance_1080P.mp4
    python3 -m dfss --url=open@sophgo.com:sophon-demo/common/dance_1080P.mp4
    mv dance_1080P.mp4 ../datasets/

    # val2017_1000.zip
    python3 -m dfss --url=open@sophgo.com:sophon-demo/common/coco_val2017_1000.tar.gz
    tar -zxf coco_val2017_1000.tar.gz -C ../datasets/
    rm coco_val2017_1000.tar.gz

    # person_keypoints_val2017_1000.json
    python3 -m dfss --url=open@sophgo.com:sophon-demo/OpenPose/datasets_0918/person_keypoints_val2017_1000.json
    mv person_keypoints_val2017_1000.json ../datasets/coco

    echo "datasets download!"
else
    echo "datasets exist!"
fi

# models
if [ ! -d "../models" ]; 
then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv8_pose/models20240730/models_BM1684.tar.gz
    tar -zxf models_BM1684.tar.gz -C ../
    rm models_BM1684.tar.gz

    python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv8_pose/models20240730/models_BM1684X.tar.gz
    tar -zxf models_BM1684X.tar.gz -C ../
    rm models_BM1684X.tar.gz

    python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv8_pose/models20240730/models_BM1688.tar.gz
    tar -zxf models_BM1688.tar.gz -C ../
    rm models_BM1688.tar.gz

    python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv8_pose/models20240730/models_CV186X.tar.gz
    tar -zxf models_CV186X.tar.gz -C ../
    rm models_CV186X.tar.gz

    python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv8_pose/models20240730/models_onnx.tar.gz
    tar -zxf models_onnx.tar.gz -C ../
    rm models_onnx.tar.gz

    python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv8_pose/models20240730/models_pt.tar.gz
    tar -zxf models_pt.tar.gz -C ../
    rm models_pt.tar.gz

    echo "models download!"
else
    echo "models exist!"
fi
popd

