#!/bin/bash
# Download TAPNext++ models and test datasets.
#   models/BM1688  -> FP16 BModels from Sophgo dfss (production precision)
#   models/onnx    -> exported ONNX graphs from Sophgo dfss
#   models/tapnextpp_ckpt.pt -> PyTorch checkpoint from Google Cloud Storage
#                               (only needed to re-export ONNX from scratch)
#   datasets/      -> test videos from Sophgo dfss
set -e

res=$(which unzip)
if [ $? != 0 ]; then
    echo "Please install unzip on your system!"
    exit 1
fi
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade

scripts_dir=$(dirname $(readlink -f "$0"))

download_bm1688=0
download_onnx=0
download_ckpt=0
download_dataset=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --BM1688)   download_bm1688=1;  shift ;;
        --onnx)     download_onnx=1;    shift ;;
        --ckpt)     download_ckpt=1;    shift ;;
        --dataset)  download_dataset=1; shift ;;
        --all)      download_bm1688=1; download_onnx=1; download_ckpt=1; download_dataset=1; shift ;;
        *) echo "Invalid option: $1" >&2; exit 1 ;;
    esac
done

pushd "$scripts_dir"
mkdir -p ../models ../datasets

# ---- datasets (test videos) ----
if [ $download_dataset -eq 1 ]; then
    if [ ! -f ../datasets/test.mp4 ]; then
        python3 -m dfss --url=open@sophgo.com:sophon-demo/TAPNextPP/datasets.zip
        unzip -o datasets.zip -d ..
        rm datasets.zip
        echo "datasets download!"
    else
        echo "datasets/test.mp4 exists; remove it to re-download."
    fi
fi

# ---- PyTorch checkpoint (~2.4 GB, from Google Cloud Storage) ----
# Only needed to re-export ONNX from scratch (see README 4.1). The exported
# ONNX is also available via --onnx, so most users can skip this.
if [ $download_ckpt -eq 1 ]; then
    if [ ! -f ../models/tapnextpp_ckpt.pt ]; then
        echo "Downloading tapnextpp_ckpt.pt from GCS ..."
        wget --no-check-certificate -q --show-progress \
            https://storage.googleapis.com/dm-tapnet/tapnextpp/tapnextpp_ckpt.pt \
            -O ../models/tapnextpp_ckpt.pt
        echo "Checkpoint saved to models/tapnextpp_ckpt.pt"
    else
        echo "models/tapnextpp_ckpt.pt exists; remove it to re-download."
    fi
fi

# ---- models ----
pushd ../models

if [ $download_bm1688 -eq 1 ]; then
    if [ ! -d BM1688 ]; then
        python3 -m dfss --url=open@sophgo.com:sophon-demo/TAPNextPP/BM1688.tar.gz
        tar xvf BM1688.tar.gz && rm BM1688.tar.gz
        echo "models/BM1688 download!"
    else
        echo "models/BM1688 folder exist! Remove it if you need to update."
    fi
fi

if [ $download_onnx -eq 1 ]; then
    if [ ! -d onnx ]; then
        python3 -m dfss --url=open@sophgo.com:sophon-demo/TAPNextPP/onnx.tar.gz
        tar xvf onnx.tar.gz && rm onnx.tar.gz
        echo "models/onnx download!"
    else
        echo "models/onnx folder exist! Remove it if you need to update."
    fi
fi

popd
popd
