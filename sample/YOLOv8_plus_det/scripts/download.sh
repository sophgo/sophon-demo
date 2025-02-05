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
        --BM1688)
            download_bm1688=1
            shift 1
            ;;
        --CV186X)
            download_cv186x=1
            shift 1
            ;;
        --onnx)
            download_onnx=1
            shift 1
            ;;
        --ckpt)
            download_ckpt=1
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
fi
    
pushd ../models

if [ ! -d "../models/BM1684X" ]; 
then
    if [ $download_bm1684x == 1 ]; then
        python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv8_plus_det/BM1684X.tar.gz
        tar xvf BM1684X.tar.gz && rm BM1684X.tar.gz
        echo "models/BM1684X download!"
    fi
else
    echo "models/BM1684X folder exist! Remove it if you need to update."
fi

if [ ! -d "../models/BM1688" ]; 
then
    if [ $download_bm1688 == 1 ]; then
        python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv8_plus_det/BM1688.tar.gz
        tar xvf BM1688.tar.gz && rm BM1688.tar.gz
        echo "models/BM1688 download!"
    fi
else
    echo "models/BM1688 folder exist! Remove it if you need to update."
fi

if [ ! -d "../models/CV186X" ]; 
then
    if [ $download_cv186x == 1 ]; then
        python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv8_plus_det/CV186X.tar.gz
        tar xvf CV186X.tar.gz && rm CV186X.tar.gz
        echo "models/CV186X download!"
    fi
else
    echo "models/BM1688 folder exist! Remove it if you need to update."
fi

if [ ! -d "../models/onnx" ]; 
then
    if [ $download_onnx == 1 ]; then
        python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv8_plus_det/onnx.tar.gz
        tar xvf onnx.tar.gz && rm onnx.tar.gz
        echo "models/onnx download!"
    fi
else
    echo "models/onnx folder exist! Remove it if you need to update."
fi
popd

popd