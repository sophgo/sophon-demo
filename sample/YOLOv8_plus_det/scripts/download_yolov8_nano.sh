#!/bin/bash
# Download yolov8-nano bmodels / onnx / qtable only (does not touch other models).
# Mirrors the ppocrv6 download_v6.sh style: a standalone script that pulls the
# prebuilt yolov8n deliverables from dfss and extracts them into models/.
#
# Usage:
#   ./scripts/download_yolov8_nano.sh
#   ./scripts/download_yolov8_nano.sh --BM1684X   # only download one platform
#   ./scripts/download_yolov8_nano.sh --BM1688
#   ./scripts/download_yolov8_nano.sh --CV186X
#   ./scripts/download_yolov8_nano.sh --onnx
#   ./scripts/download_yolov8_nano.sh --all        # default: everything
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade --quiet
scripts_dir=$(dirname $(readlink -f "$0"))

DFSS_DIR=open@sophgo.com:sophon-demo/YOLOv8_plus_det/models_yolov8_nano

download_bm1684x=0
download_bm1688=0
download_cv186x=0
download_onnx=0

if [ $# -eq 0 ]; then
    # default: download all
    download_bm1684x=1
    download_bm1688=1
    download_cv186x=1
    download_onnx=1
else
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --BM1684X) download_bm1684x=1; shift 1 ;;
            --BM1688)  download_bm1688=1;  shift 1 ;;
            --CV186X)  download_cv186x=1;   shift 1 ;;
            --onnx)    download_onnx=1;     shift 1 ;;
            --all)
                download_bm1684x=1
                download_bm1688=1
                download_cv186x=1
                download_onnx=1
                shift 1 ;;
            *)
                echo "Invalid option: $1" >&2
                echo "Usage: $0 [--BM1684X|--BM1688|--CV186X|--onnx|--all]" >&2
                exit 1 ;;
        esac
    done
fi

pushd $scripts_dir

# models folder
if [ ! -d "../models" ]; then
    mkdir -p ../models
    echo "models folder created!"
fi

# Download a per-platform tar only if the representative yolov8n bmodel is missing.
# tar is extracted into ../models/ so it merges with any existing yolov8s/9s/11s/12s files.
function dl_platform () {  # tar_name check_file
    local tar=$1 check=$2
    if [ -f "../models/$check" ]; then
        echo "$check exist! Remove it if you need to update."
    else
        pushd ../models
        python3 -m dfss --url=${DFSS_DIR}/${tar}
        tar xvf ${tar} && rm ${tar}
        popd
        echo "${tar} download!"
    fi
}

if [ $download_bm1684x -eq 1 ]; then
    dl_platform BM1684X.tar.gz BM1684X/yolov8n_fp16_1b.bmodel
fi
if [ $download_bm1688 -eq 1 ]; then
    dl_platform BM1688.tar.gz  BM1688/yolov8n_int8_4b_2core.bmodel
fi
if [ $download_cv186x -eq 1 ]; then
    dl_platform CV186X.tar.gz CV186X/yolov8n_fp16_1b.bmodel
fi

if [ $download_onnx -eq 1 ]; then
    if [ -f "../models/onnx/yolov8n.onnx" ]; then
        echo "models/onnx/yolov8n.onnx exist! Remove it if you need to update."
    else
        pushd ../models
        python3 -m dfss --url=${DFSS_DIR}/onnx.tar.gz
        tar xvf onnx.tar.gz && rm onnx.tar.gz
        popd
        echo "onnx.tar.gz download!"
    fi
fi

popd
echo "yolov8-nano download complete!"
