#!/bin/bash
#===----------------------------------------------------------------------===#
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
# 下载 GLC 例程所需的模型、派生资产与测试集。
#
# 资产布局与 README 第 3 节的目录说明一致：解开后为 models/<CHIP>/ 与 datasets/kodak24/。
# 目录已存在时跳过，需更新请先删除对应目录。
#
# 支持两个平台：BM1684X（SE7-32）与 BM1688（SE9-16）。
set -e

scripts_dir=$(dirname $(readlink -f "$0"))
top_dir=$scripts_dir/..

pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade --quiet

pushd $scripts_dir

# datasets：Kodak24 测试集
if [ ! -f "$top_dir/datasets/kodak24/kodim24.png" ]; then
    mkdir -p "$top_dir/datasets"
    pushd "$top_dir/datasets"
    python3 -m dfss --url=open@sophgo.com:sophon-demo/GLC/kodak24.tar.gz
    tar xvf kodak24.tar.gz
    rm -f kodak24.tar.gz
    popd
    echo "datasets download!"
else
    echo "Datasets folder exist! Remove it if you need to update."
fi


# models：bmodel + 派生资产（CDF 表、码本、质量向量）
# 两个平台的 bmodel 都会拉，按实际芯片用对应目录：
#   BM1684X（SE7-32）、BM1688（SE9-16）
if [ ! -d "$top_dir/models/BM1684X" ] || [ ! -d "$top_dir/models/BM1688" ]; then
    mkdir -p "$top_dir/models"
    pushd "$top_dir/models"
    for chip in BM1684X BM1688; do
        if [ -f "$chip/glc_decoder_bf16.bmodel" ]; then
            echo "models/$chip exist, skip."
            continue
        fi
        python3 -m dfss --url=open@sophgo.com:sophon-demo/GLC/$chip.tar.gz
        tar xvf $chip.tar.gz
        rm -f $chip.tar.gz
        echo "models/$chip download!"
    done
    popd
else
    echo "models folder exist! Remove it if you need to update."
fi
