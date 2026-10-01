#!/bin/bash
#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
# Download the compiled WeChatQRCode bmodels and the raw caffe source models
# from dfss.
#
# Each chip tarball holds the delivered bmodel pair:
#   detect_f32_fused.bmodel + sr_f16_fused.bmodel
# A chip-independent tarball also holds the raw caffe source models (OpenCV
# 4.8.0 WeChatQRCode, Apache-2.0), used by scripts/gen_bmodel.sh to regenerate
# the bmodels above from scratch:
#   opencv_3rdparty_wechat_qrcode.tar.gz -> models/opencv_3rdparty_wechat_qrcode/
#
# Usage:
#   scripts/download.sh [--BM1684X] [--BM1688] [--CV186X] [--all]
# (no option = download all three chips + caffe source)

scripts_dir=$(dirname $(readlink -f "$0"))

download_bm1684x=0
download_bm1688=0
download_cv186x=0

if [ $# -eq 0 ]; then
    download_bm1684x=1
    download_bm1688=1
    download_cv186x=1
fi

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
        --all)
            download_bm1684x=1
            download_bm1688=1
            download_cv186x=1
            shift 1
            ;;
        *)
            echo "Invalid option: $key" >&2
            exit 1
            ;;
    esac
done

pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade

pushd $scripts_dir

if [ ! -d "../models" ]; then
    mkdir ../models
fi

pushd ../models

if [ ! -d "BM1684X" ]; then
    if [ $download_bm1684x == 1 ]; then
        python3 -m dfss --url=open@sophgo.com:sophon-demo/wechat_qrcode/models/BM1684X.tar.gz
        tar xvf BM1684X.tar.gz && rm BM1684X.tar.gz
        echo "models/BM1684X download!"
    fi
else
    echo "models/BM1684X folder exist! Remove it if you need to update."
fi

if [ ! -d "BM1688" ]; then
    if [ $download_bm1688 == 1 ]; then
        python3 -m dfss --url=open@sophgo.com:sophon-demo/wechat_qrcode/models/BM1688.tar.gz
        tar xvf BM1688.tar.gz && rm BM1688.tar.gz
        echo "models/BM1688 download!"
    fi
else
    echo "models/BM1688 folder exist! Remove it if you need to update."
fi

if [ ! -d "CV186X" ]; then
    if [ $download_cv186x == 1 ]; then
        python3 -m dfss --url=open@sophgo.com:sophon-demo/wechat_qrcode/models/CV186X.tar.gz
        tar xvf CV186X.tar.gz && rm CV186X.tar.gz
        echo "models/CV186X download!"
    fi
else
    echo "models/CV186X folder exist! Remove it if you need to update."
fi

# Raw caffe source models (chip-independent): detect/sr prototxt + caffemodel
# from OpenCV 4.8.0 WeChatQRCode (Apache-2.0). Same tarball sophon-sail ships
# (WeChatCV/opencv_3rdparty); used by scripts/gen_bmodel.sh to regenerate the
# bmodels above from scratch.
if [ ! -d "opencv_3rdparty_wechat_qrcode" ]; then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/wechat_qrcode/models/opencv_3rdparty_wechat_qrcode.tar.gz
    tar xzf opencv_3rdparty_wechat_qrcode.tar.gz && rm opencv_3rdparty_wechat_qrcode.tar.gz
    echo "models/opencv_3rdparty_wechat_qrcode download!"
else
    echo "models/opencv_3rdparty_wechat_qrcode folder exist! Remove it if you need to update."
fi

popd
echo "models download!"
popd