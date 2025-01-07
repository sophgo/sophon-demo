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
    python3 -m dfss --url=open@sophgo.com:sophon-demo/LightStereo/cali_data.tar.gz
    tar xvf cali_data.tar.gz && rm cali_data.tar.gz
    python3 -m dfss --url=open@sophgo.com:sophon-demo/LightStereo/KITTI12.tar.gz
    tar xvf KITTI12.tar.gz && rm KITTI12.tar.gz
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
        mkdir BM1684X
        python3 -m dfss --url=open@sophgo.com:sophon-demo/LightStereo/BM1684X/LightStereo-S-SceneFlow_fp32_1b.bmodel
        python3 -m dfss --url=open@sophgo.com:sophon-demo/LightStereo/BM1684X/LightStereo-S-SceneFlow_fp16_1b.bmodel
        python3 -m dfss --url=open@sophgo.com:sophon-demo/LightStereo/BM1684X/LightStereo-S-SceneFlow_int8_1b.bmodel
        python3 -m dfss --url=open@sophgo.com:sophon-demo/LightStereo/BM1684X/LightStereo-S-SceneFlow_int8_4b.bmodel
        mv LightStereo-S-SceneFlow_*.bmodel BM1684X/
        echo "models/BM1684X download!"
    fi
else
    echo "models/BM1684X folder exist! Remove it if you need to update."
fi

if [ ! -d "../models/BM1688" ]; 
then
    if [ $download_bm1688 == 1 ]; then
        mkdir BM1688
        python3 -m dfss --url=open@sophgo.com:sophon-demo/LightStereo/BM1688/LightStereo-S-SceneFlow_fp32_1b.bmodel
        python3 -m dfss --url=open@sophgo.com:sophon-demo/LightStereo/BM1688/LightStereo-S-SceneFlow_fp16_1b.bmodel
        python3 -m dfss --url=open@sophgo.com:sophon-demo/LightStereo/BM1688/LightStereo-S-SceneFlow_fp32_1b_2core.bmodel
        python3 -m dfss --url=open@sophgo.com:sophon-demo/LightStereo/BM1688/LightStereo-S-SceneFlow_fp16_1b_2core.bmodel
        python3 -m dfss --url=open@sophgo.com:sophon-demo/LightStereo/BM1688/LightStereo-S-SceneFlow_int8_1b.bmodel
        python3 -m dfss --url=open@sophgo.com:sophon-demo/LightStereo/BM1688/LightStereo-S-SceneFlow_int8_4b.bmodel
        python3 -m dfss --url=open@sophgo.com:sophon-demo/LightStereo/BM1688/LightStereo-S-SceneFlow_int8_1b_2core.bmodel
        python3 -m dfss --url=open@sophgo.com:sophon-demo/LightStereo/BM1688/LightStereo-S-SceneFlow_int8_4b_2core.bmodel
        mv LightStereo-S-SceneFlow_*.bmodel BM1688/
        echo "models/BM1688 download!"
    fi
else
    echo "models/BM1688 folder exist! Remove it if you need to update."
fi

if [ ! -d "../models/CV186X" ]; 
then
    if [ $download_cv186x == 1 ]; then
        mkdir CV186X
        python3 -m dfss --url=open@sophgo.com:sophon-demo/LightStereo/CV186X/LightStereo-S-SceneFlow_fp32_1b.bmodel
        python3 -m dfss --url=open@sophgo.com:sophon-demo/LightStereo/CV186X/LightStereo-S-SceneFlow_fp16_1b.bmodel
        python3 -m dfss --url=open@sophgo.com:sophon-demo/LightStereo/CV186X/LightStereo-S-SceneFlow_int8_1b.bmodel
        python3 -m dfss --url=open@sophgo.com:sophon-demo/LightStereo/CV186X/LightStereo-S-SceneFlow_int8_4b.bmodel
        mv LightStereo-S-SceneFlow_*.bmodel CV186X/
        echo "models/CV186X download!"
    fi
else
    echo "models/CV186X folder exist! Remove it if you need to update."
fi

if [ ! -d "../models/onnx" ]; 
then
    if [ $download_onnx == 1 ]; then
        python3 -m dfss --url=open@sophgo.com:sophon-demo/LightStereo/onnx.tar.gz
        tar xvf onnx.tar.gz && rm onnx.tar.gz
        echo "models/onnx download!"
    fi
else
    echo "models/onnx folder exist! Remove it if you need to update."
fi

if [ ! -d "../models/ckpt" ]; 
then
    if [ $download_ckpt == 1 ]; then
        python3 -m dfss --url=open@sophgo.com:sophon-demo/LightStereo/ckpt.tar.gz
        tar xvf ckpt.tar.gz && rm ckpt.tar.gz
        echo "models/ckpt download!"
    fi
else
    echo "models/ckpt folder exist! Remove it if you need to update."
fi

popd

popd