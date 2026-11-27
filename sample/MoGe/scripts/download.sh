#!/bin/bash
# MoGe (Monocular Geometry Estimation) 例程：模型、数据集下载脚本。
#
# 编译好的 bmodel、ONNX 与评测数据集均由 SOPHON 官方上传到 DFSS。
# 支持分项下载：
#
#   ./scripts/download.sh            # 仅下载测试图片
#   ./scripts/download.sh --eval     # 下载官方评测数据集 NYUv2
#   ./scripts/download.sh --onnx     # 下载官方 ONNX 模型
#   ./scripts/download.sh --BM1688   # 下载 BM1688 平台的 bmodel
#   ./scripts/download.sh --all      # 下载以上全部

scripts_dir=$(dirname $(readlink -f "$0"))

download_onnx=0
download_bm1684x=0
download_bm1688=0
download_cv186x=0
download_eval=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --onnx)   download_onnx=1;   shift 1;;
        --BM1684X) download_bm1684x=1; shift 1;;
        --BM1688) download_bm1688=1; shift 1;;
        --CV186X) download_cv186x=1; shift 1;;
        --eval)   download_eval=1;   shift 1;;
        --all)
            download_onnx=1; download_eval=1
            download_bm1684x=1; download_bm1688=1; download_cv186x=1
            shift 1;;
        *)
            echo "Invalid option: $1" >&2; exit 1;;
    esac
done

pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade 2>/dev/null

pushd "$scripts_dir"

# ---------- 测试图片(来自 MoGe 官方 example_images) ----------
if [ ! -d "../datasets/test" ]; then
    mkdir -p ../datasets
    pushd ../datasets
    python3 -m dfss --url=open@sophgo.com:sophon-demo/MoGev2/test.tar.gz
    tar xvf test.tar.gz && rm test.tar.gz
    popd
    echo "datasets/test download!"
else
    echo "datasets/test exists! Remove it if you need to update."
fi

# ---------- 官方评测数据集 NYUv2(Ruicheng/monocular-geometry-evaluation) ----------
if [ $download_eval == 1 ]; then
    if [ ! -d "../datasets/NYUv2" ]; then
        mkdir -p ../datasets
        pushd ../datasets
        python3 -m dfss --url=open@sophgo.com:sophon-demo/MoGev2/NYUv2.tar.gz
        tar xvf NYUv2.tar.gz && rm NYUv2.tar.gz
        popd
        echo "datasets/NYUv2 download!"
    else
        echo "datasets/NYUv2 exists! Remove it if you need to update."
    fi
fi

# ---------- 官方 ONNX 模型 ----------
if [ $download_onnx == 1 ]; then
    if [ ! -d "../models/onnx" ]; then
        mkdir -p ../models
        pushd ../models
        python3 -m dfss --url=open@sophgo.com:sophon-demo/MoGev2/onnx.tar.gz
        tar xvf onnx.tar.gz && rm onnx.tar.gz
        popd
        echo "models/onnx download!"
    else
        echo "models/onnx exists! Remove it if you need to update."
    fi
fi

# ---------- 编译好的 bmodel (SOPHON 官方 DFSS) ----------
if [ $download_bm1684x == 1 ] && [ ! -d "../models/BM1684X" ]; then
    mkdir -p ../models
    pushd ../models
    python3 -m dfss --url=open@sophgo.com:sophon-demo/MoGev2/BM1684X.tar.gz
    tar xvf BM1684X.tar.gz && rm BM1684X.tar.gz
    popd
    echo "models/BM1684X download!"
fi
if [ $download_bm1688 == 1 ] && [ ! -d "../models/BM1688" ]; then
    mkdir -p ../models
    pushd ../models
    python3 -m dfss --url=open@sophgo.com:sophon-demo/MoGev2/BM1688.tar.gz
    tar xvf BM1688.tar.gz && rm BM1688.tar.gz
    popd
    echo "models/BM1688 download!"
fi
if [ $download_cv186x == 1 ] && [ ! -d "../models/CV186X" ]; then
    mkdir -p ../models
    pushd ../models
    python3 -m dfss --url=open@sophgo.com:sophon-demo/MoGev2/CV186X.tar.gz
    tar xvf CV186X.tar.gz && rm CV186X.tar.gz
    popd
    echo "models/CV186X download!"
fi

popd
