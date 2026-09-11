#!/bin/bash
# ==============================================================================
# 下载测试数据集与模型 (YOLO-World v2)
# 用法: ./scripts/download.sh [--BM1684X|--BM1684X2|--onnx|--all]
# 数据来源与 sample/YOLO_world 一致 (sophgo dfss)
# ==============================================================================
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
scripts_dir=$(dirname $(readlink -f "$0"))

download_bm1684x=0
download_bm1684x2=0
download_onnx=0

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --BM1684X)
            download_bm1684x=1
            shift 1
            ;;
        --BM1684X2)
            download_bm1684x2=1
            shift 1
            ;;
        --onnx)
            download_onnx=1
            shift 1
            ;;
        --all)
            download_bm1684x=1
            download_bm1684x2=1
            download_onnx=1
            shift 1
            ;;
        *)
            # 无参数时默认下载 BM1684X 与数据集，保持向后兼容
            if [ "$#" -eq 0 ] || [ "$1" == "" ]; then
                download_bm1684x=1
            else
                echo "Invalid option: $key" >&2
                exit 1
            fi
            shift 1
            ;;
    esac
done

pushd "$scripts_dir"
# datasets
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
    python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLO_world_v2/cali_npz.tar.gz
    tar xvf cali_npz.tar.gz && rm cali_npz.tar.gz
    popd
    echo "datasets download!"
else
    echo "Datasets folder exist! Remove it if you need to update."
fi

# models
if [ ! -d "../models" ]; then
    mkdir ../models
fi

pushd ../models

# CLIP 分词与 text_projection 等公共文件
if [ ! -f "../models/bpe_simple_vocab_16e6.txt.gz" ]; then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLO_world_v2/bpe_simple_vocab_16e6.txt.gz
fi
if [ ! -f "../models/text_projection_512_512.npy" ]; then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLO_world_v2/text_projection_512_512.npy
fi

if [ ! -d "../models/BM1684X" ];
then
    if [ $download_bm1684x == 1 ]; then
        python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLO_world_v2/BM1684X.tar.gz
        tar xvf BM1684X.tar.gz && rm BM1684X.tar.gz
        echo "models/BM1684X download!"
    fi
else
    echo "models/BM1684X folder exist! Remove it if you need to update."
fi

if [ ! -d "../models/BM1684X2" ];
then
    if [ $download_bm1684x2 == 1 ]; then
        python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLO_world_v2/BM1684X2.tar.gz
        tar xvf BM1684X2.tar.gz && rm BM1684X2.tar.gz
        echo "models/BM1684X2 download!"
    fi
else
    echo "models/BM1684X2 folder exist! Remove it if you need to update."
fi

if [ ! -d "../models/onnx" ];
then
    if [ $download_onnx == 1 ]; then
        python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLO_world_v2/onnx.tar.gz
        tar xvf onnx.tar.gz && rm onnx.tar.gz
        echo "models/onnx download!"
    fi
else
    echo "models/onnx folder exist! Remove it if you need to update."
fi
popd

popd
