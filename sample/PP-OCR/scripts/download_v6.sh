#!/bin/bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
scripts_dir=$(dirname $(readlink -f "$0"))

pushd $scripts_dir

# datasets
if [ ! -d "../datasets" ];
then
    mkdir -p ../datasets
    echo "datasets folder created!"
else
    echo "Datasets folder exist! Remove it if you need to update."
fi

# models
if [ ! -d "../models" ];
then
    mkdir -p ../models
    echo "models folder created!"
fi

# Download v6 dict files
pushd ../datasets
if [ ! -f "ppocrv6_dict.txt" ]; then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/PP-OCR/models_v6/ppocrv6_dict.txt
    echo "ppocrv6_dict.txt download!"
else
    echo "ppocrv6_dict.txt exist! Remove it if you need to update."
fi
if [ ! -f "ppocrv6_tiny_dict.txt" ]; then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/PP-OCR/models_v6/ppocrv6_tiny_dict.txt
    echo "ppocrv6_tiny_dict.txt download!"
else
    echo "ppocrv6_tiny_dict.txt exist! Remove it if you need to update."
fi
popd

# Download v6 bmodel files
pushd ../models
# BM1688 (含 1core 和 2core)
if [ ! -d "BM1688" ]; then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/PP-OCR/models_v6/BM1688.tar.gz
    tar xvf BM1688.tar.gz && rm BM1688.tar.gz
    echo "BM1688 models download!"
else
    echo "BM1688 folder exist! Remove it if you need to update."
fi

# CV186X
if [ ! -d "CV186X" ]; then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/PP-OCR/models_v6/CV186X.tar.gz
    tar xvf CV186X.tar.gz && rm CV186X.tar.gz
    echo "CV186X models download!"
else
    echo "CV186X folder exist! Remove it if you need to update."
fi

# BM1684X
if [ ! -d "BM1684X" ]; then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/PP-OCR/models_v6/BM1684X.tar.gz
    tar xvf BM1684X.tar.gz && rm BM1684X.tar.gz
    echo "BM1684X models download!"
else
    echo "BM1684X folder exist! Remove it if you need to update."
fi
popd

popd
echo "PP-OCRv6 download complete!"
