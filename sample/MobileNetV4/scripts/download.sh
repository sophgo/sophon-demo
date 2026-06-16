#!/bin/bash
scripts_dir=$(dirname $(readlink -f "$0"))

pushd $scripts_dir
# datasets
if [ ! -d "../datasets/imagenet_val_1k" ];
then
    echo "Downloading datasets..."
    pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
    python3 -m dfss --url=open@sophgo.com:sophon-demo/ResNet/datasets_0918/datasets.zip
    unzip datasets.zip -d ../
    rm datasets.zip
    echo "datasets download!"
else
    echo "Datasets folder exist! Remove it if you need to update."
fi

# models
if [ ! -d "../models" ];
then
    echo "Models folder not found. Please compile models first or download pre-compiled models."
    echo "Run: ./scripts/gen_fp32bmodel_mlir.sh <target>"
    echo "     ./scripts/gen_fp16bmodel_mlir.sh <target>"
    echo "     ./scripts/gen_int8bmodel_mlir.sh <target>"
else
    echo "Models folder exist! Remove it if you need to update."
fi
popd
