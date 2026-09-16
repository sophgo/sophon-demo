#!/bin/bash
# download.sh — 下载 π0.5 例程所需的 bmodel 与数据集
#
# 用法: ./download.sh [all|bm1684x]
#
# 前置: pip3 install dfss
# 资产托管在 open@sophgo.com:sophon-demo/Pi0_5/ 下（上传清单见 docs/dfss_upload_manifest.md）

pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
scripts_dir=$(dirname $(readlink -f "$0"))

pushd $scripts_dir

function download_datasets {
    if [ ! -d "../datasets" ]; then
        mkdir -p ../datasets
    fi
    pushd ../datasets
    # 固定 seed 的观测样本 + 官方动作真值（npy），用于"单次推理"例程的精度测试
    if [ ! -d "pi05_libero_sample" ]; then
        python3 -m dfss --url=open@sophgo.com:sophon-demo/Pi0_5/datasets/pi05_libero_sample.tar.gz
        tar xvf pi05_libero_sample.tar.gz && rm pi05_libero_sample.tar.gz
    fi
    popd
}

function download_bm1684x {
    if [ ! -d "../models/BM1684X" ]; then
        mkdir -p ../models/BM1684X
    fi
    pushd ../models/BM1684X
    # 6 个子模型 bmodel，合计约 3.45 GB
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Pi0_5/models/BM1684X.tar.gz
    tar xvf BM1684X.tar.gz && rm BM1684X.tar.gz
    popd
}

if [ "$1" == "bm1684x" ]; then
    download_datasets
    download_bm1684x
elif [ "$1" == "all" ]; then
    download_datasets
    download_bm1684x
else
    echo "Error Parameter"
    echo "Usage: $0 [all|bm1684x]"
    exit 1
fi

pushd ..
echo "下载完成。models/BM1684X 与 datasets/ 目录如下："
ls -la models/BM1684X
ls -la datasets
popd
