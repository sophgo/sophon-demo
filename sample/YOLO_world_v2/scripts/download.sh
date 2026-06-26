#!/bin/bash
# ==============================================================================
# 下载测试数据集 (YOLO-World v2)
# bmodel 已由 scripts/gen_*bmodel_mlir.sh 本地编译, 无需下载
# 数据来源与 sample/YOLO_world 一致 (sophgo dfss)
# ==============================================================================
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
scripts_dir=$(dirname $(readlink -f "$0"))
pushd "$scripts_dir"

if [ ! -d "../datasets" ]; then
    mkdir ../datasets
    pushd ../datasets
    python3 -m dfss --url=open@sophgo.com:sophon-demo/common/test.tar.gz
    tar xvf test.tar.gz && rm test.tar.gz
    python3 -m dfss --url=open@sophgo.com:sophon-demo/common/coco.names
    python3 -m dfss --url=open@sophgo.com:sophon-demo/common/coco128.tar.gz
    tar xvf coco128.tar.gz && rm coco128.tar.gz
    python3 -m dfss --url=open@sophgo.com:sophon-demo/common/coco_val2017_1000.tar.gz
    tar xvf coco_val2017_1000.tar.gz && rm coco_val2017_1000.tar.gz
    python3 -m dfss --url=open@sophgo.com:sophon-demo/common/test_car_person_1080P.mp4
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
    python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLO_world_v2/models/BM1684X.tar.gz
    tar xvf BM1684X.tar.gz && rm BM1684X.tar.gz
    python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLO_world_v2/models/onnx.tar.gz
    tar xvf onnx.tar.gz && rm onnx.tar.gz
    python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLO_world_v2/models/text_projection_512_512.npy
    python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLO_world_v2/models/bpe_simple_vocab_16e6.txt.gz
    popd
    echo "models download!"
else
    echo "Models folder exist! Remove it if you need to update."
fi

popd
